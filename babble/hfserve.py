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

import os
import time
from dataclasses import dataclass
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


def _require_hf_sampling():
    """Sampling helpers, imported lazily with the rest of the optional extra."""
    try:
        from transformers.generation.logits_process import (
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )
    except ImportError as exc:
        raise HFServeError(
            "BABBLE_SERVE_BACKEND=hf needs the optional hf extra -- "
            'install with `pip install -e ".[hf]"`'
        ) from exc
    return TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper


@dataclass(frozen=True)
class HFGenerationStats:
    """Low-overhead timing and token counts from one HF generation."""

    ttft_s: float
    total_s: float
    steady_s: float
    candidate_token_counts: tuple[int, ...]
    first_step_tokens: int
    selected_index: int
    selected_content_tokens: int


class _CandidateTracker:
    """Apply Babble's extra penalties and score best-of without score history.

    Transformers' ``output_scores=True`` retains a ``batch × vocab`` tensor
    for every generated step, then stacks the whole history to recover one
    scalar per chosen token. This object is both the final logits processor
    and the streamer: it normalizes the current step, then gathers only the
    tokens the sampler actually chose. Peak score storage is therefore one
    ``batch × vocab`` step plus ``steps × batch`` scalars rather than
    ``steps × batch × vocab`` scores.

    Temperature/top-k/top-p live here as well so scoring observes the exact
    post-warp distribution used for sampling. Built-in repetition and n-gram
    processors still run before this object.
    """

    def __init__(
        self,
        *,
        frequency_penalty: float,
        presence_penalty: float,
        temperature: float,
        top_k: int,
        top_p: float,
        pad_id: int,
        eos_id: int,
    ) -> None:
        Temperature, TopK, TopP = _require_hf_sampling()
        self.frequency_penalty = float(frequency_penalty)
        self.presence_penalty = float(presence_penalty)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)
        self.warpers = []
        if temperature != 1.0:
            self.warpers.append(Temperature(float(temperature)))
        if top_k:
            self.warpers.append(TopK(int(top_k), min_tokens_to_keep=1))
        if top_p < 1.0:
            self.warpers.append(TopP(float(top_p), min_tokens_to_keep=1))
        self._pending_logprobs: torch.Tensor | None = None
        self._token_logprobs: list[torch.Tensor] = []
        self._counts: list[int] | None = None
        self.started_s = 0.0
        self.first_token_s: float | None = None
        self.last_token_s: float | None = None
        self.first_step_tokens = 0

    def start(self, started_s: float | None = None) -> None:
        self.started_s = time.perf_counter() if started_s is None else started_s

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.frequency_penalty or self.presence_penalty:
            counts = torch.zeros_like(scores)
            counts.scatter_add_(
                1,
                input_ids,
                torch.ones_like(input_ids, dtype=scores.dtype),
            )
            scores.sub_(counts * self.frequency_penalty)
            if self.presence_penalty:
                scores.sub_((counts > 0).to(scores.dtype) * self.presence_penalty)
        for warper in self.warpers:
            scores = warper(input_ids, scores)
        # Keep only batch-sized scalars between processor and streamer calls.
        self._pending_logprobs = torch.log_softmax(scores, dim=-1)
        return scores

    def put(self, value: torch.Tensor) -> None:
        # generate() streams the expanded prompt once before sampled tokens.
        if value.ndim > 1:
            batch = int(value.shape[0])
            self._counts = [0] * batch
            return
        if self._pending_logprobs is None or self._counts is None:
            return
        now = time.perf_counter()
        if self.first_token_s is None:
            self.first_token_s = now
            self.first_step_tokens = int((value != self.pad_id).sum().item())
        self.last_token_s = now
        chosen = value.to(self._pending_logprobs.device)
        selected = self._pending_logprobs.gather(1, chosen[:, None]).squeeze(1)
        live = value != self.pad_id
        self._token_logprobs.append(selected.masked_fill(~live.to(selected.device), 0.0))
        for i, token in enumerate(value.tolist()):
            if int(token) != self.pad_id:
                self._counts[i] += 1
        self._pending_logprobs = None

    def end(self) -> None:
        pass

    def result(self, *, generated: torch.Tensor) -> tuple[int, list[int], HFGenerationStats]:
        if self._counts is None or not self._token_logprobs:
            raise HFServeError("Transformers generation did not stream token scores")
        totals = torch.stack(self._token_logprobs).sum(dim=0)
        means = totals / torch.tensor(
            self._counts,
            dtype=totals.dtype,
            device=totals.device,
        ).clamp_min(1)
        means = means.masked_fill(
            torch.tensor(self._counts, device=totals.device) == 0,
            float("-inf"),
        )
        best_idx = int(means.argmax())
        now = time.perf_counter()
        first = self.first_token_s or now
        last = self.last_token_s or first
        keep = [
            int(token)
            for token in generated[best_idx]
            if int(token) not in (self.pad_id, self.eos_id)
        ]
        return best_idx, keep, HFGenerationStats(
            ttft_s=max(0.0, first - self.started_s),
            total_s=max(0.0, now - self.started_s),
            steady_s=max(0.0, last - first),
            candidate_token_counts=tuple(self._counts),
            first_step_tokens=self.first_step_tokens,
            selected_index=best_idx,
            selected_content_tokens=len(keep),
        )


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
        self.model_id = model_dir.name
        self.param_count = sum(p.numel() for p in self.model.parameters())
        # The HF backend historically ignored these native-sampler controls.
        # Applying Settings' non-zero default would silently change the live
        # model's output distribution, so corrected support is explicitly
        # gated until it has its own quality evaluation.
        self._extra_penalties = os.environ.get(
            "BABBLE_HF_FREQUENCY_PENALTIES", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        # The bot's exchange records carry a training step; a hand-promoted HF
        # model has no meaningful one, so it serves as step 0 by convention.
        self.step = 0
        self.log.event(
            "model.load",
            source="hf",
            model_dir=str(model_dir),
            step=self.step,
            params=self.param_count,
            device="cpu",
            frequency_presence_penalties=self._extra_penalties,
        )

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        # Leave room for the reply and the two structural tokens.
        budget = self.config.max_position_embeddings - self.settings.max_new_tokens - 2
        if budget <= 0:
            raise HFServeError(
                "max_new_tokens leaves no room for an HF prompt: "
                f"context={self.config.max_position_embeddings}, "
                f"max_new_tokens={self.settings.max_new_tokens}"
            )
        if len(tokens) > budget:
            tokens = tokens[-budget:]
        return torch.tensor([[self.bos_id, *tokens, self.sep_id]], dtype=torch.long)

    def conversation_prompt(
        self,
        history,
        current_user: str,
        *,
        max_turns: int,
        max_tokens: int,
        max_chars: int,
    ) -> str:
        """Serialize complete turns within this model's exact token budget."""
        from .conversation import conversation_prompt_for_token_budget

        architectural_budget = (
            self.config.max_position_embeddings - self.settings.max_new_tokens - 2
        )
        return conversation_prompt_for_token_budget(
            history,
            current_user,
            max_turns=max_turns,
            max_chars=max_chars,
            max_tokens=min(
                max(1, architectural_budget),
                max(1, int(max_tokens)),
            ),
            token_count=lambda text: len(
                self.tokenizer.encode(text, add_special_tokens=False).ids
            ),
        )

    def _generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        best_of: int,
    ) -> tuple[str, HFGenerationStats]:
        s = self.settings
        started = time.perf_counter()
        tracker = _CandidateTracker(
            frequency_penalty=s.frequency_penalty if self._extra_penalties else 0.0,
            presence_penalty=s.presence_penalty if self._extra_penalties else 0.0,
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
        )
        tracker.start(started)
        input_ids = self._encode_prompt(prompt).to(self.device)
        with torch.inference_mode():
            sequences = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=max(1, int(best_of)),
                max_new_tokens=max(1, int(max_new_tokens)),
                # The tracker applies these after the built-in repetition and
                # n-gram processors, then records the same warped distribution
                # the sampler consumes. Keeping the built-in warpers disabled
                # prevents applying them twice.
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=s.repetition_penalty,
                no_repeat_ngram_size=s.no_repeat_ngram_size or None,
                eos_token_id=self.eos_id,
                pad_token_id=self.pad_id,
                use_cache=True,
                logits_processor=[tracker],
                streamer=tracker,
            )
        generated = sequences[:, input_ids.shape[1] :]
        _best_idx, keep, stats = tracker.result(generated=generated)
        return self.tokenizer.decode(keep, skip_special_tokens=True).strip(), stats

    def __call__(self, prompt: str) -> Generation:
        s = self.settings
        started = time.perf_counter()
        text, stats = self._generate(
            prompt,
            max_new_tokens=s.max_new_tokens,
            best_of=s.best_of,
        )
        return Generation(
            text=text,
            step=self.step,
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
            repetition_penalty=s.repetition_penalty,
            frequency_penalty=s.frequency_penalty if self._extra_penalties else 0.0,
            presence_penalty=s.presence_penalty if self._extra_penalties else 0.0,
            no_repeat_ngram_size=s.no_repeat_ngram_size,
            max_new_tokens=s.max_new_tokens,
            ms=(time.perf_counter() - started) * 1000,
        )

    def benchmark_sample(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        best_of: int,
    ) -> HFGenerationStats:
        """Run a bounded sample against this already-loaded serving model."""
        _text, stats = self._generate(
            prompt,
            max_new_tokens=max_new_tokens,
            best_of=best_of,
        )
        return stats

    def benchmark_metadata(self) -> dict[str, object]:
        """Serving facts reported alongside the bounded benchmark run."""
        dtype = next(self.model.parameters()).dtype
        return {
            "model": self.model_id,
            "params": self.param_count,
            "dtype": str(dtype).removeprefix("torch."),
            "optimizations": (
                "native CPU default device",
                "KV cache",
                "streamed best-of scores",
                "frequency/presence penalties on" if self._extra_penalties else "frequency/presence penalties off",
            ),
        }


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
