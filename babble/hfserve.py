"""Serve a HuggingFace Transformers model instead of a native Babbler checkpoint.

This exists for the promoted `ProCreations/Booper-Big-Chat-INT8` line: a small
Mixtral-architecture MoE chat model, quantized to INT8 on disk, trained on the
same BPE tokenizer (vocab 16384) and the same pair layout
(`<bos> prompt <sep> response <eos>`) as the native checkpoints. It cannot be
expressed as a `Babbler`, so it gets its own generator behind the same
one-callable seam `core.Babble` already accepts.

Deliberate differences from `CheckpointGenerator`:

* **Local files only.** `Settings.hf_model_dir` points at an on-disk snapshot
  (fetched once at deploy time); serving never touches the Hub.
* **No hot reload.** There is no trainer writing this model mid-flight; it is
  promoted by hand, so it loads once at startup and stays loaded.
* **Always pair layout.** These are chat-SFT models; `serve_layout` does not
  apply. Feeding one a bare continuation would waste what it was trained on.
* **Optional dependency.** `transformers`/`safetensors` are the `hf` extra in
  pyproject; nothing outside this module imports them, so the native path
  keeps its deliberately tiny footprint.

The INT8 artifact stores every matrix as int8 plus a per-output-channel scale
(`<name>.scale`); norms stay floating point. `_load_int8` re-expands that into
an ordinary Mixtral state dict, handling both expert layouts Transformers has
shipped (per-expert `block_sparse_moe.experts.N.w{1,2,3}` and the newer fused
`mlp.experts.gate_up_proj`/`down_proj` stacks).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .config import Settings
from .core import Generation
from .cpu_runtime import configure_cpu, force_cpu_device
from .logs import EventLog, NullLog


class HFServeError(RuntimeError):
    """The HF backend cannot start: missing extra, missing dir, bad snapshot."""


def _require_hf():
    try:
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from transformers import MixtralConfig, MixtralForCausalLM
    except ImportError as exc:
        raise HFServeError(
            "BABBLE_SERVE_BACKEND=hf needs the optional hf extra -- "
            'install with `pip install -e ".[hf]"`'
        ) from exc
    return load_file, Tokenizer, MixtralConfig, MixtralForCausalLM


def _unpack(packed: dict, name: str) -> torch.Tensor:
    value = packed[name]
    if value.dtype == torch.int8:
        value = value.to(torch.float32) * packed[name + ".scale"].to(torch.float32)
    return value.to(torch.float32)


def _load_int8(model_dir: Path):
    """The INT8 snapshot as an eval-mode fp32 Mixtral on CPU.

    fp32 rather than the artifact's nominal bf16 because this always serves on
    CPU, where fp32 matmuls are the fast path and bf16 is emulated.
    """
    load_file, _, MixtralConfig, MixtralForCausalLM = _require_hf()
    weights_path = model_dir / "model-int8.safetensors"
    if not weights_path.exists():
        raise HFServeError(f"no model-int8.safetensors under {model_dir}")
    packed = load_file(str(weights_path))
    config = MixtralConfig.from_pretrained(model_dir)
    model = MixtralForCausalLM(config)

    state: dict[str, torch.Tensor] = {}
    for name in packed:
        if name.endswith(".scale") or ".block_sparse_moe." in name:
            continue
        state[name] = _unpack(packed, name)

    # The artifact always stores experts per-expert under `block_sparse_moe`;
    # what the installed Transformers expects depends on its version.
    fused = any(".mlp.experts.gate_up_proj" in k for k in model.state_dict())
    for layer in range(config.num_hidden_layers):
        old = f"model.layers.{layer}.block_sparse_moe"
        new = f"model.layers.{layer}.mlp" if fused else old
        gate_name = (new if fused else old) + ".gate.weight"
        state[gate_name] = _unpack(packed, old + ".gate.weight")
        if fused:
            gate_up, down = [], []
            for expert in range(config.num_local_experts):
                prefix = f"{old}.experts.{expert}"
                gate_up.append(
                    torch.cat(
                        (_unpack(packed, prefix + ".w1.weight"), _unpack(packed, prefix + ".w3.weight")),
                        dim=0,
                    )
                )
                down.append(_unpack(packed, prefix + ".w2.weight"))
            state[new + ".experts.gate_up_proj"] = torch.stack(gate_up)
            state[new + ".experts.down_proj"] = torch.stack(down)
        else:
            for expert in range(config.num_local_experts):
                prefix = f"{old}.experts.{expert}"
                for w in ("w1", "w2", "w3"):
                    state[f"{prefix}.{w}.weight"] = _unpack(packed, f"{prefix}.{w}.weight")

    missing, unexpected = model.load_state_dict(state, strict=False)
    # `lm_head.weight` is tied to the embedding and legitimately absent.
    if unexpected or any(name != "lm_head.weight" for name in missing):
        raise HFServeError(f"snapshot/model mismatch: missing={missing}, unexpected={unexpected}")
    model.tie_weights()
    return model.eval(), config


class HFGenerator:
    """Callable the bot hands to `Babble`, backed by a local HF snapshot."""

    def __init__(self, settings: Settings, log: EventLog | None = None) -> None:
        self.settings = settings
        self.log = log or NullLog()
        if settings.hf_model_dir is None:
            raise HFServeError("BABBLE_SERVE_BACKEND=hf but BABBLE_HF_MODEL_DIR is not set")
        model_dir = settings.hf_model_dir
        if not model_dir.is_dir():
            raise HFServeError(f"BABBLE_HF_MODEL_DIR={model_dir} is not a directory")
        configure_cpu(getattr(settings, "infer_threads", None) or settings.train_threads)
        self.device = force_cpu_device()

        _, Tokenizer, _, _ = _require_hf()
        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise HFServeError(f"no tokenizer.json under {model_dir}")
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        ids = {name: self.tokenizer.token_to_id(name) for name in ("<pad>", "<bos>", "<sep>", "<eos>")}
        absent = [name for name, tid in ids.items() if tid is None]
        if absent:
            raise HFServeError(f"tokenizer at {tok_path} lacks special tokens: {absent}")
        self.pad_id, self.bos_id, self.sep_id, self.eos_id = (
            ids["<pad>"],
            ids["<bos>"],
            ids["<sep>"],
            ids["<eos>"],
        )

        self.model, self.config = _load_int8(model_dir)
        self.model.to(self.device)
        # The bot's exchange records carry a training step; a hand-promoted HF
        # model has no meaningful one, so it serves as step 0 by convention.
        self.step = 0
        self.log.event(
            "model.load",
            source="hf",
            model_dir=str(model_dir),
            step=self.step,
            params=sum(p.numel() for p in self.model.parameters()),
            device="cpu",
        )

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        # Leave room for the reply and the two structural tokens.
        budget = self.config.max_position_embeddings - self.settings.max_new_tokens - 2
        if len(tokens) > budget:
            tokens = tokens[-budget:]
        return torch.tensor([[self.bos_id, *tokens, self.sep_id]], dtype=torch.long)

    def __call__(self, prompt: str) -> Generation:
        s = self.settings
        started = time.perf_counter()
        input_ids = self._encode_prompt(prompt).to(self.device)
        n = max(1, s.best_of)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=n,
                max_new_tokens=s.max_new_tokens,
                temperature=s.temperature,
                top_k=s.top_k,
                top_p=s.top_p,
                repetition_penalty=s.repetition_penalty,
                no_repeat_ngram_size=s.no_repeat_ngram_size or None,
                eos_token_id=self.eos_id,
                pad_token_id=self.pad_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        # Same best-of rule as the native path: keep the candidate the model
        # itself finds most likely, by mean per-token logprob so short replies
        # aren't favoured just for stopping early.
        generated = out.sequences[:, input_ids.shape[1] :]
        scores = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        best_idx, best_score = 0, None
        for i in range(generated.shape[0]):
            token_ids = generated[i]
            live = token_ids != self.pad_id
            count = int(live.sum())
            if count == 0:
                continue
            mean = float(scores[i][live].sum()) / count
            if best_score is None or mean > best_score:
                best_idx, best_score = i, mean
        keep = [int(t) for t in generated[best_idx] if int(t) not in (self.pad_id, self.eos_id)]
        text = self.tokenizer.decode(keep, skip_special_tokens=True).strip()
        return Generation(
            text=text,
            step=self.step,
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
            repetition_penalty=s.repetition_penalty,
            no_repeat_ngram_size=s.no_repeat_ngram_size,
            max_new_tokens=s.max_new_tokens,
            ms=(time.perf_counter() - started) * 1000,
        )


def make_generator(settings: Settings, log: EventLog | None = None):
    """The generator `Settings.serve_backend` names.

    Imports stay inside the branches so the checkpoint path never pays for
    (or requires) transformers.
    """
    if settings.serve_backend == "hf":
        return HFGenerator(settings, log)
    if settings.serve_backend == "checkpoint":
        from .generate import CheckpointGenerator

        return CheckpointGenerator(settings, log)
    raise ValueError(
        f"unknown serve_backend {settings.serve_backend!r} -- expected 'checkpoint' or 'hf'"
    )
