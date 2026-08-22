# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1",
#   "datasets>=2.19,<3",
#   "huggingface_hub>=0.23",
#   "pyarrow>=14",
# ]
# ///
"""Stage 1: pretrain booper's architecture on a bounded, streamed slice of an
English web corpus (openbmb/Ultra-FineWeb-L1) on a real GPU.

INTENTIONALLY SELF-CONTAINED. This file does not import anything from the
`babble` package, on purpose: it is meant to be handed to someone who owns
compute we don't have and run with a single command --

    hf jobs uv run --flavor l4x1 --timeout 6h --secrets HF_TOKEN \\
        https://raw.githubusercontent.com/kowo-co/babble/main/pretrain_hf.py \\
        -- --config configs/pretrain/default.json --output-dir /data/pretrain

-- with zero repo checkout and no dependency on `babble` being installed in
the job's container. `uv run` reads the `# /// script` block above and
installs exactly those four packages; nothing else has to exist on the
machine that runs this.

The cost of that is real: `ModelConfig`/`Babbler` below are a deliberate,
line-for-line mirror of `babble/model.py`, and `train_bpe`/`BPETokenizer`
mirror the BPE half of `babble/subword.py`. They are NOT re-exports -- if you
change the architecture in `babble/model.py`, mirror the change here too, or
a checkpoint this script produces will silently stop matching what
`babble/posttrain.py` expects to load. What is NOT duplicated is the
checkpoint *format*: this writes `{"step", "config", "model", "optim", ...}`
payloads and a companion `tokenizer.json` (just the ordered merge list --
`BPETokenizer.vocab` is fully determined by replaying merges over the 256 raw
bytes, so nothing else needs to round-trip) that `babble.subword.BPETokenizer`
and `babble.model.Babbler`/`ModelConfig` load directly, no translation step.
See `HF_PRETRAIN_PIPELINE.md` for the full design writeup, dataset choice,
model-size justification, and cost/wall-clock estimates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- config -----------------------------------------------------------------


@dataclass
class PretrainConfig:
    # Dataset. Six Common Crawl snapshots exist as separate HF dataset
    # "configs" (config_name literally "CC-MAIN-2025-30" etc, confirmed
    # against the dataset's own README metadata) -- train and val are
    # deliberately different snapshots so held-out loss can never leak train
    # phrasing just by virtue of both being "the same web corpus".
    dataset_id: str = "openbmb/Ultra-FineWeb-L1"
    train_config: str = "CC-MAIN-2025-51"
    val_config: str = "CC-MAIN-2025-47"
    text_field: str = "content"
    # Total *trainable* tokens (mask==1 positions, i.e. everything but each
    # example's opening <bos>) to consume before stopping. This is the knob
    # that bounds "a large amount but not an insane amount" -- streamed, so
    # raising it never changes how much lands on disk, only how long the job
    # runs. See HF_PRETRAIN_PIPELINE.md for the token budget vs model size
    # (Chinchilla-style) reasoning behind each preset's default.
    token_budget: int = 600_000_000
    val_docs: int = 512
    # How many train-config documents to read (once, up front) to fit the
    # BPE tokenizer. Bounded independently of the training budget: fitting is
    # pure-Python and its cost is O(unique chunks), not O(corpus size), but it
    # is still a one-time up-front cost worth capping explicitly.
    tokenizer_fit_docs: int = 20_000

    # Model shape (config, not code -- every field here is a CLI/JSON knob).
    vocab_size: int = 16384
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1

    # Optimisation.
    batch_size: int = 64
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Run mechanics.
    seed: int = 1337
    checkpoint_every_steps: int = 200
    keep_checkpoints: int = 3
    log_every_steps: int = 20
    amp_bf16: bool = True

    # Optional durability: push checkpoint + tokenizer + loss log to a HF Hub
    # model repo after every checkpoint. Strongly recommended on HF Jobs --
    # the job's filesystem is deleted when the job ends (see "Persist your
    # results" in the HF Jobs docs) -- but off by default so a local
    # smoke-test run never needs a token or touches the network.
    hub_checkpoint_repo: str = ""
    hub_private: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "PretrainConfig":
        raw = json.loads(path.read_text())
        known = {f: raw[f] for f in cls.__dataclass_fields__ if f in raw}
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            print(f"[config] ignoring unknown keys in {path}: {unknown}", file=sys.stderr)
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)


# --- model: mirrors babble/model.py -----------------------------------------


@dataclass
class ModelConfig:
    vocab_size: int = 260
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        known = {f: raw[f] for f in cls.__dataclass_fields__ if f in raw}
        return cls(**known)


def _drop(p: float) -> nn.Module:
    return nn.Dropout(p) if p > 0.0 else nn.Identity()


class KVCache:
    __slots__ = ("k", "v", "length", "max_len")

    def __init__(self, n_layer, batch, n_head, head_dim, max_len, *, device, dtype=torch.float32):
        self.max_len = max_len
        self.length = 0
        self.k = [torch.zeros(batch, n_head, max_len, head_dim, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(batch, n_head, max_len, head_dim, device=device, dtype=dtype) for _ in range(n_layer)]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must divide evenly into n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        drop = self.dropout if self.training else 0.0
        if cache is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:
            start = cache.length
            end = start + T
            if end > cache.max_len:
                raise ValueError(f"KV cache overflow: need {end} slots, have {cache.max_len}")
            cache.k[self.layer_idx][:, :, start:end, :] = k
            cache.v[self.layer_idx][:, :, start:end, :] = v
            k_full = cache.k[self.layer_idx][:, :, :end, :]
            v_full = cache.v[self.layer_idx][:, :, :end, :]
            if start == 0:
                out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True, dropout_p=drop)
            elif T == 1:
                out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False, dropout_p=drop)
            else:
                rows = torch.arange(start, end, device=q.device)[:, None]
                cols = torch.arange(end, device=q.device)[None, :]
                allow = cols <= rows
                out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=allow, dropout_p=drop)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            _drop(config.dropout),
        )

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache=cache)
        return x + self.mlp(self.ln2(x))


class Babbler(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.drop = _drop(c.dropout)
        self.blocks = nn.ModuleList(Block(c, layer_idx=i) for i in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def new_cache(self, batch_size: int = 1, max_len: int | None = None) -> KVCache:
        c = self.config
        head_dim = c.n_embd // c.n_head
        weight = self.tok_emb.weight
        slots = c.block_size if max_len is None else min(int(max_len), c.block_size)
        return KVCache(c.n_layer, batch_size, c.n_head, head_dim, slots, device=weight.device, dtype=weight.dtype)

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        _, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence of {T} exceeds block_size {self.config.block_size}")
        if cache is None:
            pos = torch.arange(T, device=idx.device)
        else:
            start = cache.length
            if start + T > self.config.block_size:
                raise ValueError(f"sequence of {start + T} exceeds block_size {self.config.block_size}")
            pos = torch.arange(start, start + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, cache=cache)
        if cache is not None:
            cache.length += T
        return self.lm_head(self.ln_f(x))

    def num_params(self) -> int:
        seen = {id(p): p.numel() for p in self.parameters()}
        return sum(seen.values())


def sequence_loss(model: Babbler, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view_as(targets)
    scale = mask[:, 1:].to(per_token.dtype)
    return (per_token * scale).sum() / scale.sum().clamp(min=1e-8)


# --- tokenizer: mirrors the BPE half of babble/subword.py -------------------
#
# Byte-level BPE, same scheme as babble.subword.BPETokenizer: ids 0..255 are
# raw bytes, 256.. are learned merges in the order they were learned, and the
# four specials (pad/bos/sep/eos) sit just above the learned vocab. The
# `tokenizer.json` this writes is exactly what `BPETokenizer.from_merges`
# (added to babble/subword.py alongside this script) needs to reconstruct an
# identical tokenizer -- `vocab` is fully determined by replaying `merges`
# over the 256 raw bytes, so only the merge list has to round-trip.

_CHUNK_RE = re.compile(r"\s+|\S+")


def _chunks(text: str) -> list[str]:
    return _CHUNK_RE.findall(text)


def _pair_counts(word_ids: dict[str, list[int]], freq: dict[str, int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for chunk, ids in word_ids.items():
        f = freq[chunk]
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + f
    return counts


def _merge_ids(ids: list[int], a: int, b: int, new_id: int) -> list[int]:
    if len(ids) < 2:
        return ids
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(texts: list[str], num_merges: int) -> list[tuple[int, int, int]]:
    """Learn `num_merges` byte-pair merges from `texts`. Returns the ordered
    merge list; the vocab (id -> bytes) is derivable from it alone."""
    freq: dict[str, int] = {}
    for text in texts:
        for chunk in _chunks(text):
            freq[chunk] = freq.get(chunk, 0) + 1
    word_ids = {chunk: list(chunk.encode("utf-8")) for chunk in freq}

    merges: list[tuple[int, int, int]] = []
    next_id = 256
    for i in range(num_merges):
        counts = _pair_counts(word_ids, freq)
        if not counts:
            break
        (a, b) = max(counts, key=lambda p: (counts[p], p))
        new_id = next_id
        next_id += 1
        merges.append((a, b, new_id))
        for chunk in word_ids:
            word_ids[chunk] = _merge_ids(word_ids[chunk], a, b, new_id)
        if (i + 1) % 1000 == 0:
            print(f"[tokenizer] {i + 1}/{num_merges} merges learned", flush=True)
    return merges


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int, int]]) -> None:
        self.merges = merges
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for a, b, new_id in merges:
            vocab[new_id] = vocab[a] + vocab[b]
        self.vocab = vocab
        base = 256 + len(merges)
        self.pad, self.bos, self.sep, self.eos = base, base + 1, base + 2, base + 3

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + 4

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        for a, b, new_id in self.merges:
            ids = _merge_ids(ids, a, b, new_id)
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _chunks(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytearray()
        for i in ids:
            piece = self.vocab.get(i)
            if piece is not None:
                raw.extend(piece)
        return bytes(raw).decode("utf-8", errors="replace")

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({"merges": [list(m) for m in self.merges]}))

    @classmethod
    def from_json(cls, path: Path) -> "BPETokenizer":
        raw = json.loads(path.read_text())
        return cls([tuple(m) for m in raw["merges"]])


@dataclass(frozen=True)
class Example:
    tokens: list[int]
    mask: list[int]

    def __len__(self) -> int:
        return len(self.tokens)


def text_examples(tok: BPETokenizer, text: str, block_size: int) -> list[Example]:
    """`<bos> chunk [<eos>]`, chunked to fit `block_size`; everything after
    <bos> is a target. Identical layout to `babble.subword.text_examples`."""
    budget = block_size - 2
    if budget < 1:
        raise ValueError(f"block_size {block_size} too small")
    ids = tok.encode(text)
    if not ids:
        return []
    chunks = [ids[i : i + budget] for i in range(0, len(ids), budget)]
    out = []
    for index, chunk in enumerate(chunks):
        final = index == len(chunks) - 1
        tokens = [tok.bos, *chunk] + ([tok.eos] if final else [])
        mask = [0] + [1] * (len(tokens) - 1)
        out.append(Example(tokens=tokens, mask=mask))
    return out


def stack_examples(examples: list[Example], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(e.tokens) for e in examples)
    tokens = torch.full((len(examples), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e.tokens)
        tokens[i, :n] = torch.as_tensor(e.tokens, dtype=torch.long)
        mask[i, :n] = torch.as_tensor(e.mask, dtype=torch.long)
    return tokens.to(device), mask.to(device)


@torch.inference_mode()
def sample(model: Babbler, tok: BPETokenizer, prompt: str, device, max_new_tokens=60, temperature=0.7, top_k=40) -> str:
    was_training = model.training
    model.eval()
    try:
        ids = [tok.bos, *tok.encode(prompt)]
        ids = ids[-model.config.block_size :]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        cache = model.new_cache(batch_size=1, max_len=min(len(ids) + max_new_tokens, model.config.block_size))
        logits = model(idx, cache=cache)
        out_ids = list(ids)
        for _ in range(max_new_tokens):
            last = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(last, min(top_k, last.size(-1)))
                last[last < v[:, [-1]]] = -float("inf")
            probs = F.softmax(last, dim=-1)
            next_id = int(torch.multinomial(probs, 1))
            if next_id == tok.eos or cache.length >= cache.max_len:
                break
            out_ids.append(next_id)
            logits = model(torch.tensor([[next_id]], dtype=torch.long, device=device), cache=cache)
        return tok.decode(out_ids[1:])
    finally:
        model.train(was_training)


# --- device -------------------------------------------------------------


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# --- checkpointing --------------------------------------------------------


def save_checkpoint(
    ckpt_dir: Path,
    model: Babbler,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_consumed: int,
    docs_consumed: int,
    loss: float,
    keep: int,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scratch = ckpt_dir / ".partial"
    scratch.mkdir(exist_ok=True)
    payload = {
        "step": step,
        "tokens_consumed": tokens_consumed,
        "docs_consumed": docs_consumed,
        "loss": loss,
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    archive = ckpt_dir / f"ckpt-{step:08d}.pt"
    tmp = scratch / f"{archive.name}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, archive)
    latest_tmp = scratch / "latest.pt.tmp"
    torch.save(payload, latest_tmp)
    os.replace(latest_tmp, ckpt_dir / "latest.pt")
    # Prune old numbered checkpoints, keeping `latest.pt` and the newest `keep`.
    archives = sorted(ckpt_dir.glob("ckpt-*.pt"))
    for old in archives[:-keep] if keep > 0 else []:
        old.unlink(missing_ok=True)
    return archive


def maybe_push_to_hub(cfg: PretrainConfig, ckpt_dir: Path, tokenizer_path: Path) -> None:
    if not cfg.hub_checkpoint_repo:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[hub] hub_checkpoint_repo set but $HF_TOKEN is empty -- skipping push", file=sys.stderr)
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(cfg.hub_checkpoint_repo, private=cfg.hub_private, repo_type="model", exist_ok=True)
        for name in ("latest.pt", "loss.jsonl"):
            p = ckpt_dir / name
            if p.exists():
                api.upload_file(path_or_fileobj=str(p), path_in_repo=name, repo_id=cfg.hub_checkpoint_repo, repo_type="model")
        if tokenizer_path.exists():
            api.upload_file(
                path_or_fileobj=str(tokenizer_path),
                path_in_repo=tokenizer_path.name,
                repo_id=cfg.hub_checkpoint_repo,
                repo_type="model",
            )
        print(f"[hub] pushed to {cfg.hub_checkpoint_repo}", flush=True)
    except Exception as exc:  # network/auth/rate-limit -- never fatal, the job keeps training
        print(f"[hub] push failed ({type(exc).__name__}: {exc}) -- continuing", file=sys.stderr)


# --- data streaming -------------------------------------------------------


def open_stream(dataset_id: str, config_name: str, docs_to_skip: int = 0):
    from datasets import load_dataset

    ds = load_dataset(dataset_id, name=config_name, split="train", streaming=True)
    if docs_to_skip:
        ds = ds.skip(docs_to_skip)
    return iter(ds)


def collect_docs(dataset_id: str, config_name: str, text_field: str, n: int) -> list[str]:
    stream = open_stream(dataset_id, config_name)
    out = []
    for doc in stream:
        text = doc.get(text_field) or ""
        if text:
            out.append(text)
        if len(out) >= n:
            break
    return out


# --- lr schedule -----------------------------------------------------------


def lr_at(step: int, cfg: PretrainConfig, total_steps: int) -> float:
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    span = max(1, total_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    floor = cfg.min_lr_ratio
    return cfg.learning_rate * (floor + (1 - floor) * cosine)


# --- main training loop ----------------------------------------------------


def train(cfg: PretrainConfig, output_dir: Path, device: torch.device, max_steps: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    loss_path = ckpt_dir / "loss.jsonl"

    torch.manual_seed(cfg.seed)

    # --- tokenizer: fit once, reused on every resume -----------------------
    if tokenizer_path.exists():
        tok = BPETokenizer.from_json(tokenizer_path)
        print(f"[tokenizer] loaded {tok.vocab_size}-token vocab from {tokenizer_path}", flush=True)
    else:
        num_merges = max(0, cfg.vocab_size - 256 - 4)
        print(f"[tokenizer] fitting {num_merges} BPE merges on {cfg.tokenizer_fit_docs} docs "
              f"from {cfg.dataset_id}/{cfg.train_config} ...", flush=True)
        fit_texts = collect_docs(cfg.dataset_id, cfg.train_config, cfg.text_field, cfg.tokenizer_fit_docs)
        merges = train_bpe(fit_texts, num_merges)
        tok = BPETokenizer(merges)
        tok.to_json(tokenizer_path)
        print(f"[tokenizer] fit {tok.vocab_size}-token vocab, saved to {tokenizer_path}", flush=True)

    # --- held-out val set: a DIFFERENT CC-MAIN snapshot than train ---------
    val_cache_path = output_dir / "val_examples.pt"
    if val_cache_path.exists():
        val_examples = torch.load(val_cache_path, weights_only=False)
    else:
        print(f"[val] collecting {cfg.val_docs} docs from {cfg.val_config} (disjoint snapshot)...", flush=True)
        val_texts = collect_docs(cfg.dataset_id, cfg.val_config, cfg.text_field, cfg.val_docs)
        val_examples = [ex for t in val_texts for ex in text_examples(tok, t, cfg.block_size)]
        torch.save(val_examples, val_cache_path)
    print(f"[val] {len(val_examples)} held-out examples", flush=True)

    # --- model / optimizer: fresh, or resumed from latest.pt ---------------
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        payload = torch.load(latest, map_location=device, weights_only=False)
        model_cfg = ModelConfig.from_dict(payload["config"])
        model = Babbler(model_cfg).to(device)
        model.load_state_dict(payload["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        optimizer.load_state_dict(payload["optim"])
        step = payload["step"]
        tokens_consumed = payload["tokens_consumed"]
        docs_consumed = payload["docs_consumed"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        print(f"[resume] step {step}, {tokens_consumed:,}/{cfg.token_budget:,} tokens, "
              f"{docs_consumed:,} docs already consumed", flush=True)
    else:
        model_cfg = ModelConfig(
            vocab_size=tok.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
            n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout,
        )
        model = Babbler(model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        step = 0
        tokens_consumed = 0
        docs_consumed = 0
        print(f"[model] fresh init, {model.num_params():,} params, config={model_cfg}", flush=True)

    total_steps = max(1, cfg.token_budget // max(1, cfg.batch_size * cfg.block_size))
    if max_steps is not None:
        total_steps = min(total_steps, step + max_steps)

    stop_requested = False

    def _handle_signal(signum, frame):
        nonlocal stop_requested
        print(f"[signal] {signum} received, finishing current step then checkpointing", flush=True)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    stream = open_stream(cfg.dataset_id, cfg.train_config, docs_to_skip=docs_consumed)
    example_buffer: list[Example] = []
    use_amp = device.type == "cuda" and cfg.amp_bf16
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _nullcontext()

    window: list[float] = []
    t_start = time.time()
    print(f"[train] budget {cfg.token_budget:,} tokens (~{total_steps:,} steps at "
          f"batch {cfg.batch_size} x block {cfg.block_size}), device={device}", flush=True)

    while tokens_consumed < cfg.token_budget and not stop_requested:
        while len(example_buffer) < cfg.batch_size:
            try:
                doc = next(stream)
            except StopIteration:
                print(f"[train] stream exhausted for config {cfg.train_config} -- stopping short of budget", flush=True)
                tokens_consumed = cfg.token_budget  # force the outer loop to stop
                break
            docs_consumed += 1
            text = doc.get(cfg.text_field) or ""
            if text:
                example_buffer.extend(text_examples(tok, text, cfg.block_size))
        if tokens_consumed >= cfg.token_budget:
            break

        batch = example_buffer[: cfg.batch_size]
        example_buffer = example_buffer[cfg.batch_size :]
        tokens, mask = stack_examples(batch, tok.pad, device)

        lr = lr_at(step, cfg, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        with amp_ctx:
            loss = sequence_loss(model, tokens, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        step += 1
        batch_tokens = int(mask[:, 1:].sum())
        tokens_consumed += batch_tokens
        window.append(float(loss.item()))

        if step % cfg.log_every_steps == 0:
            elapsed = time.time() - t_start
            tok_per_s = tokens_consumed / max(elapsed, 1e-6)
            print(
                f"[step {step:7d}] loss {sum(window) / len(window):7.4f} | lr {lr:.2e} | "
                f"tokens {tokens_consumed:>12,}/{cfg.token_budget:,} | {tok_per_s:9.1f} tok/s",
                flush=True,
            )
            window = []

        if step % cfg.checkpoint_every_steps == 0 or tokens_consumed >= cfg.token_budget or stop_requested:
            val_loss = eval_val(model, val_examples, tok.pad, device, amp_ctx)
            samples = {
                p: sample(model, tok, p, device, max_new_tokens=40)
                for p in ("the cat", "In the beginning", "Scientists have discovered", "The weather today is")
            }
            entry = {
                "step": step,
                "tokens_consumed": tokens_consumed,
                "docs_consumed": docs_consumed,
                "train_loss": round(sum(window) / len(window), 6) if window else None,
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
                "lr": lr,
                "elapsed_s": round(time.time() - t_start, 1),
                "tokens_per_s": round(tokens_consumed / max(time.time() - t_start, 1e-6), 1),
                "samples": samples,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with open(loss_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            print(f"[checkpoint] step {step} | val_loss {val_loss} | sample('the cat') -> {samples['the cat']!r}", flush=True)
            save_checkpoint(ckpt_dir, model, optimizer, step, tokens_consumed, docs_consumed, entry["train_loss"] or float("nan"), cfg.keep_checkpoints)
            maybe_push_to_hub(cfg, ckpt_dir, tokenizer_path)

        if stop_requested:
            break

    print(f"[done] step {step}, tokens_consumed {tokens_consumed:,}/{cfg.token_budget:,}, "
          f"elapsed {time.time() - t_start:.1f}s", flush=True)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@torch.inference_mode()
def eval_val(model: Babbler, val_examples: list[Example], pad_id: int, device, amp_ctx, chunk: int = 32) -> float | None:
    if not val_examples:
        return None
    was_training = model.training
    model.eval()
    try:
        total_loss = 0.0
        total_tokens = 0
        for i in range(0, len(val_examples), chunk):
            batch = val_examples[i : i + chunk]
            tokens, mask = stack_examples(batch, pad_id, device)
            with amp_ctx:
                logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
            ).view_as(targets)
            scale = mask[:, 1:].to(per_token.dtype)
            total_loss += float((per_token * scale).sum())
            total_tokens += float(scale.sum())
        return total_loss / max(total_tokens, 1e-8)
    finally:
        model.train(was_training)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON PretrainConfig (see configs/pretrain/*.json)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where checkpoints/tokenizer.json/loss.jsonl go")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--token-budget", type=int, default=None, help="Override config token_budget")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many additional steps (debug/smoke-test)")
    args = parser.parse_args(argv)

    cfg = PretrainConfig.from_json(args.config)
    if args.token_budget is not None:
        cfg.token_budget = args.token_budget

    device = resolve_device(args.device)
    print(f"[device] {device} (cuda available: {torch.cuda.is_available()})", flush=True)
    train(cfg, args.output_dir, device, args.max_steps)


if __name__ == "__main__":
    main()
