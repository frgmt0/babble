# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1",
#   "datasets>=2.19,<3",
#   "huggingface_hub>=0.23",
#   "pyarrow>=14",
# ]
# ///
"""Self-contained GPU continue-pretrain of booper on Discord-Dialogues.

Loads the existing 600M-token weights (trained on Ultra-FineWeb-L1) and
continues on Discord chat. Does not import the `babble` package. Model /
tokenizer code mirrors `babble.model` / `babble.subword` so the checkpoint
loads in the serving bot. Prefer `./run.sh` over invoking this file directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import shutil
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DEFAULT_START_CKPT = HERE / "start.pt"
SHAPE_FIELDS = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")
LOSS_KEYS = (
    "step",
    "loss",
    "tokens_seen",
    "docs_consumed",
    "elapsed_s",
    "tokens_per_s",
    "wall_clock",
    "threads",
    "batch_size",
    "block_size",
)


@dataclass
class RunConfig:
    dataset_repo: str = "mookiezi/Discord-Dialogues"
    dataset_revision: str = "a8b2294bd5b4acfe4ce537b688e7eee111c50fe2"
    dataset_path: str = "data/train.parquet"
    dataset_sha256: str = "241e350e7f651085c5c2cb4d5274f7cb671b84b3d5fba091101823678da454ec"
    dataset_size_bytes: int = 346784147
    tokenizer: str = "tokenizer.json"
    vocab_size: int = 16384
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    token_budget: int = 1_000_000_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337
    batch_size: int = 64
    amp_bf16: bool = True
    workers: int = 4
    log_every: int = 1
    checkpoint_every: int = 50
    keep_checkpoints: int = 3

    @classmethod
    def from_json(cls, path: Path) -> "RunConfig":
        raw = json.loads(path.read_text())
        known = {f.name: raw[f.name] for f in fields(cls) if f.name in raw}
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            print(f"[config] ignoring unknown keys: {unknown}", file=sys.stderr)
        return cls(**known)


# --- model (mirrors babble/model.py) --------------------------------------


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
        known = {f.name: raw[f.name] for f in fields(cls) if f.name in raw}
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
        return KVCache(
            c.n_layer, batch_size, c.n_head, head_dim, slots, device=weight.device, dtype=weight.dtype
        )

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        _, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence of {T} exceeds block_size {self.config.block_size}")
        if cache is None:
            pos = torch.arange(T, device=idx.device)
        else:
            start = cache.length
            if start + T > self.config.block_size:
                raise ValueError(
                    f"sequence of {start + T} exceeds block_size {self.config.block_size}"
                )
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


# --- tokenizer (mirrors babble/subword.py BPE half) -----------------------

_CHUNK_RE = re.compile(r"\s+|\S+")


def _chunks(text: str) -> list[str]:
    return _CHUNK_RE.findall(text)


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


@dataclass
class Example:
    tokens: list[int]
    mask: list[int]


def text_examples(tok: BPETokenizer, text: str, block_size: int) -> list[Example]:
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


def stack_examples(examples: list[Example], pad_id: int, device: torch.device):
    width = max(len(e.tokens) for e in examples)
    tokens = torch.full((len(examples), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e.tokens)
        tokens[i, :n] = torch.as_tensor(e.tokens, dtype=torch.long)
        mask[i, :n] = torch.as_tensor(e.mask, dtype=torch.long)
    return tokens.to(device), mask.to(device)


# --- data -----------------------------------------------------------------


def row_to_text(row: dict) -> str:
    for key in ("text", "content", "conversation", "conversations", "messages"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list) and val:
            parts = []
            for turn in val:
                if isinstance(turn, str) and turn.strip():
                    parts.append(turn)
                elif isinstance(turn, dict):
                    role = turn.get("role") or turn.get("author") or ""
                    body = turn.get("content") or turn.get("text") or turn.get("value") or ""
                    if body:
                        parts.append(f"{role}: {body}" if role else str(body))
            if parts:
                return "\n".join(parts)
    blobs = [v for v in row.values() if isinstance(v, str) and v.strip()]
    return "\n".join(blobs)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_parquet(cfg: RunConfig, cache_dir: Path) -> Path:
    dest = cache_dir / "train.parquet"
    if dest.exists() and dest.stat().st_size == cfg.dataset_size_bytes:
        digest = sha256_file(dest)
        if digest.lower() == cfg.dataset_sha256.lower():
            print(f"[data] using cached {dest} (sha256 ok)", flush=True)
            return dest
        print("[data] cache checksum mismatch, re-downloading", flush=True)
        dest.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[data] downloading {cfg.dataset_repo}@{cfg.dataset_revision} "
        f"{cfg.dataset_path} ({cfg.dataset_size_bytes / 1e6:.0f} MB) ...",
        flush=True,
    )
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=cfg.dataset_repo,
        filename=cfg.dataset_path,
        repo_type="dataset",
        revision=cfg.dataset_revision,
        local_dir=str(cache_dir / "hf"),
    )
    src = Path(downloaded)
    shutil.copy2(src, dest)
    digest = sha256_file(dest)
    if digest.lower() != cfg.dataset_sha256.lower():
        raise RuntimeError(
            f"checksum mismatch for {dest}: got {digest}, expected {cfg.dataset_sha256}"
        )
    print(f"[data] verified sha256 {digest}", flush=True)
    return dest


def parquet_num_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def epoch_and_offset(docs_consumed: int, n_rows: int) -> tuple[int, int]:
    if n_rows <= 0:
        return 0, 0
    return divmod(max(0, int(docs_consumed)), n_rows)


def iter_parquet_rows(path: Path, skip: int = 0):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    seen = 0
    for batch in pf.iter_batches(batch_size=256):
        rows = batch.to_pylist()
        for row in rows:
            if seen < skip:
                seen += 1
                continue
            yield row


def iter_parquet_shuffled(path: Path, skip: int, seed: int, epoch: int):
    """One shuffled epoch of the parquet, skipping the first `skip` perm slots."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    n = table.num_rows
    g = torch.Generator()
    g.manual_seed(int(seed) + int(epoch))
    perm = torch.randperm(n, generator=g)
    if skip:
        perm = perm[skip:]
    if perm.numel() == 0:
        return
        yield  # make this a generator even when empty
    shuffled = table.take(pa.array(perm.cpu().numpy()))
    for batch in shuffled.to_batches(max_chunksize=256):
        yield from batch.to_pylist()


def corpus_docs_offset(ckpt_dataset_id: str | None, docs_consumed: int, target_repo: str) -> int:
    """How many Discord parquet rows to skip on resume.

    Skip is only valid when the checkpoint's `dataset_id` is this same
    Discord corpus (mid-run `latest.pt`). The 600M-token start.pt was
    trained on Ultra-FineWeb-L1 — its 408,581 docs_consumed indexes that
    web stream, not Discord-Dialogues. Using it as a parquet skip would
    throw away ~408k unseen Discord rows.
    """
    if ckpt_dataset_id != target_repo:
        return 0
    return max(0, int(docs_consumed))


def iter_corpus(path: Path, docs_consumed: int, seed: int):
    """Yield rows forever, advancing through epochs.

    `docs_consumed` here is the count of Discord-Dialogues rows already
    used on *this* corpus (0 when starting from the Ultra-FineWeb
    checkpoint). Epoch 0 is file order of the pinned parquet. Epoch 1+
    uses `randperm(seed + epoch)` over the full table (loads parquet into
    RAM). The default 1B-token budget stays inside epoch 0 (~7.3M rows).
    """
    n = parquet_num_rows(path)
    if n <= 0:
        return
    while True:
        epoch, offset = epoch_and_offset(docs_consumed, n)
        print(
            f"[data] epoch {epoch} skip {offset:,}/{n:,} rows "
            f"(docs_consumed={docs_consumed:,})",
            flush=True,
        )
        if epoch == 0:
            source = iter_parquet_rows(path, skip=offset)
        else:
            source = iter_parquet_shuffled(path, skip=offset, seed=seed, epoch=epoch)
        yielded = 0
        for row in source:
            yielded += 1
            yield row
        docs_consumed += yielded
        if yielded == 0:
            if offset == 0:
                raise RuntimeError(f"parquet at {path} produced zero rows")
            docs_consumed = (epoch + 1) * n


def iter_jsonl_rows(path: Path, skip: int = 0):
    seen = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if seen < skip:
                seen += 1
                continue
            yield json.loads(line)


def prefetch_rows(source, workers: int):
    """Tiny prefetch so parquet decode overlaps with the GPU step."""
    if workers <= 0:
        yield from source
        return
    q: queue.Queue = queue.Queue(maxsize=max(8, workers * 4))
    sentinel = object()

    def _run():
        try:
            for item in source:
                q.put(item)
        finally:
            q.put(sentinel)

    threading.Thread(target=_run, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


# --- checkpoint / log -----------------------------------------------------


def utcnow_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


class VocabMismatch(RuntimeError):
    pass


class CheckpointError(RuntimeError):
    """Start checkpoint missing, unreadable, or architecture mismatch."""


def assert_shape_matches(ckpt_config: dict, cfg: RunConfig) -> None:
    mismatches = []
    for key in SHAPE_FIELDS:
        got = ckpt_config.get(key)
        want = getattr(cfg, key)
        if got != want:
            mismatches.append(f"{key}: checkpoint={got!r} training={want!r}")
    if mismatches:
        raise CheckpointError(
            "checkpoint architecture does not match training config:\n  "
            + "\n  ".join(mismatches)
        )


def assert_vocab_matches(model: Babbler, tok: BPETokenizer) -> None:
    if int(model.config.vocab_size) != int(tok.vocab_size):
        raise VocabMismatch(
            f"model vocab_size={model.config.vocab_size} != tokenizer {tok.vocab_size}"
        )


def save_checkpoint(
    ckpt_dir: Path,
    model: Babbler,
    optimizer: torch.optim.Optimizer,
    tok: BPETokenizer,
    step: int,
    tokens_consumed: int,
    docs_consumed: int,
    loss: float,
    keep: int,
    dataset_id: str,
) -> Path:
    assert_vocab_matches(model, tok)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scratch = ckpt_dir / ".partial"
    scratch.mkdir(exist_ok=True)
    payload = {
        "step": step,
        "loss": loss,
        "tokens_consumed": tokens_consumed,
        "docs_consumed": docs_consumed,
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "saved_at": utcnow_iso(),
        "dataset_id": dataset_id,
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        payload["cuda_rng"] = torch.cuda.get_rng_state()
    archive = ckpt_dir / f"ckpt-{step:08d}.pt"
    tmp = scratch / f"{archive.name}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, archive)
    latest_tmp = scratch / "latest.pt.tmp"
    shutil.copyfile(archive, latest_tmp)
    os.replace(latest_tmp, ckpt_dir / "latest.pt")
    archives = sorted(ckpt_dir.glob("ckpt-*.pt"))
    for old in archives[:-keep] if keep > 0 else []:
        old.unlink(missing_ok=True)
    return archive


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def load_payload(path: Path, device: torch.device, cfg: RunConfig):
    print(f"[resume] loading {path}", flush=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_config = payload.get("config") or {}
    assert_shape_matches(ckpt_config, cfg)
    model_cfg = ModelConfig.from_dict(ckpt_config)
    model = Babbler(model_cfg).to(device)
    model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    if "optim" not in payload:
        raise CheckpointError(f"{path} has no optimizer state (`optim`); refusing to guess")
    optimizer.load_state_dict(payload["optim"])
    step = int(payload["step"])
    tokens_consumed = int(payload.get("tokens_consumed", 0))
    raw_docs = int(payload.get("docs_consumed", 0))
    ckpt_dataset = payload.get("dataset_id")
    docs_consumed = corpus_docs_offset(ckpt_dataset, raw_docs, cfg.dataset_repo)
    if docs_consumed != raw_docs:
        print(
            f"[resume] checkpoint dataset_id={ckpt_dataset!r} is not "
            f"{cfg.dataset_repo}; starting Discord-Dialogues at row 0 "
            f"(ignoring web-corpus docs_consumed={raw_docs:,}). "
            f"Keeping tokens_consumed={tokens_consumed:,} and optimizer.",
            flush=True,
        )
    if "torch_rng" in payload:
        torch.set_rng_state(payload["torch_rng"].cpu())
    if "cuda_rng" in payload and device.type == "cuda":
        torch.cuda.set_rng_state(payload["cuda_rng"].cpu())
    print(
        f"[resume] step {step}, tokens {tokens_consumed:,}, "
        f"discord_docs {docs_consumed:,} from {path}",
        flush=True,
    )
    return model, optimizer, step, tokens_consumed, docs_consumed


def train(
    cfg: RunConfig,
    output_dir: Path,
    device: torch.device,
    *,
    smoke: bool,
    from_scratch: bool,
    checkpoint: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loss_path = output_dir / "loss.jsonl"
    tok_src = HERE / cfg.tokenizer
    tok_dest = output_dir / "tokenizer.json"
    if not tok_src.is_file():
        raise FileNotFoundError(f"tokenizer not shipped at {tok_src}")
    if not tok_dest.exists() or tok_dest.read_bytes() != tok_src.read_bytes():
        shutil.copy2(tok_src, tok_dest)
    tok = BPETokenizer.from_json(tok_dest)
    if tok.vocab_size != cfg.vocab_size:
        raise VocabMismatch(
            f"tokenizer vocab_size={tok.vocab_size} != config {cfg.vocab_size}"
        )
    print(f"[tokenizer] {tok.vocab_size}-token BPE from {tok_dest}", flush=True)

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    out_latest = ckpt_dir / "latest.pt"
    start_path = checkpoint if checkpoint is not None else DEFAULT_START_CKPT

    if smoke and from_scratch:
        use_scratch = True
        load_path = None
    elif from_scratch and not out_latest.exists():
        use_scratch = True
        load_path = None
    elif out_latest.exists() and not from_scratch:
        use_scratch = False
        load_path = out_latest
    elif from_scratch:
        use_scratch = True
        load_path = None
    elif start_path.is_file():
        use_scratch = False
        load_path = start_path
    else:
        raise CheckpointError(
            f"start checkpoint not found at {start_path}.\n"
            "Copy the ~409MB latest.pt you were sent to gpu-pretrain/start.pt,\n"
            "or pass --checkpoint /path/to/latest.pt.\n"
            "Refusing to initialize from scratch (pass --from-scratch if you "
            "really want random weights)."
        )

    if use_scratch:
        if not smoke and not from_scratch:
            raise CheckpointError("internal: scratch init without --from-scratch")
        model_cfg = ModelConfig(
            vocab_size=tok.vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
        )
        model = Babbler(model_cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        step = 0
        tokens_consumed = 0
        docs_consumed = 0
        print(f"[model] fresh init, {model.num_params():,} params", flush=True)
    else:
        model, optimizer, step, tokens_consumed, docs_consumed = load_payload(
            load_path, device, cfg
        )

    assert_vocab_matches(model, tok)

    if not smoke and tokens_consumed >= cfg.token_budget:
        print(
            f"[done] checkpoint already at {tokens_consumed:,} tokens; "
            f"budget is {cfg.token_budget:,}. Raise --token-budget to continue.",
            flush=True,
        )
        print(f"[send-back] {ckpt_dir / 'latest.pt'} and {tok_dest}", flush=True)
        return

    max_steps = None
    batch_size = cfg.batch_size
    workers = cfg.workers
    token_budget = cfg.token_budget
    if smoke:
        max_steps = 20
        batch_size = 1
        workers = 0
        token_budget = 50_000
        print("[smoke] 20 steps, batch=1, bundled smoke.jsonl, no parquet download", flush=True)
        rows = iter_jsonl_rows(HERE / "smoke.jsonl", skip=0)
        # Cycle the tiny slice so 20 steps always have text.
        def _cycle():
            while True:
                yielded = 0
                for row in iter_jsonl_rows(HERE / "smoke.jsonl", skip=0):
                    yielded += 1
                    yield row
                if yielded == 0:
                    raise RuntimeError("smoke.jsonl is empty")

        rows = _cycle()
    else:
        parquet = ensure_parquet(cfg, HERE / ".cache")
        rows = prefetch_rows(iter_corpus(parquet, docs_consumed, cfg.seed), workers)

    stop_requested = False

    def _handle(signum, _frame):
        nonlocal stop_requested
        print(f"[signal] {signum} — finish step then checkpoint", flush=True)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    use_amp = device.type == "cuda" and cfg.amp_bf16
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _nullcontext()
    print(
        f"[train] budget {token_budget:,} batch={batch_size}x{cfg.block_size} "
        f"device={device} amp_bf16={use_amp} workers={workers}",
        flush=True,
    )

    buffer: list[Example] = []
    window: list[float] = []
    t0 = time.time()
    tokens_at_start = tokens_consumed
    steps_this_proc = 0
    threads = max(1, int(workers) if device.type == "cuda" else torch.get_num_threads())
    row_iter = iter(rows)

    while tokens_consumed < token_budget and not stop_requested:
        while len(buffer) < batch_size:
            try:
                row = next(row_iter)
            except StopIteration:
                print("[train] corpus iterator ended", flush=True)
                break
            docs_consumed += 1
            text = row_to_text(row)
            if text:
                buffer.extend(text_examples(tok, text, cfg.block_size))
        if tokens_consumed >= token_budget:
            break
        if not buffer:
            break

        batch = buffer[:batch_size]
        buffer = buffer[batch_size:]
        tokens, mask = stack_examples(batch, tok.pad, device)

        model.train()
        with amp_ctx:
            loss = sequence_loss(model, tokens, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        step += 1
        steps_this_proc += 1
        batch_tokens = int(mask[:, 1:].sum())
        tokens_consumed += batch_tokens
        window.append(float(loss.item()))

        elapsed = time.time() - t0
        run_tokens = max(1, tokens_consumed - tokens_at_start)
        tok_s = run_tokens / max(elapsed, 1e-6)
        mean_loss = sum(window) / len(window)

        if step % cfg.log_every == 0:
            entry = {
                "step": step,
                "loss": round(mean_loss, 6),
                "tokens_seen": tokens_consumed,
                "docs_consumed": docs_consumed,
                "elapsed_s": round(elapsed, 2),
                "tokens_per_s": round(tok_s, 2),
                "wall_clock": utcnow_iso(),
                "threads": threads,
                "batch_size": batch_size,
                "block_size": cfg.block_size,
            }
            log_jsonl(loss_path, entry)
            print(
                f"[step {step:7d}] loss {mean_loss:7.4f} | "
                f"tokens {tokens_consumed:,} | {tok_s:7.1f} tok/s",
                flush=True,
            )
            window = []

        do_ckpt = (
            (not smoke and step % cfg.checkpoint_every == 0)
            or tokens_consumed >= token_budget
            or stop_requested
            or (smoke and max_steps is not None and steps_this_proc >= max_steps)
        )
        if do_ckpt:
            save_checkpoint(
                ckpt_dir,
                model,
                optimizer,
                tok,
                step,
                tokens_consumed,
                docs_consumed,
                mean_loss,
                cfg.keep_checkpoints,
                cfg.dataset_repo,
            )
            print(f"[checkpoint] step {step} tokens {tokens_consumed:,}", flush=True)

        if max_steps is not None and steps_this_proc >= max_steps:
            break

    print(
        f"[done] step {step} tokens {tokens_consumed:,}/{token_budget:,} "
        f"elapsed {time.time() - t0:.1f}s",
        flush=True,
    )
    print(
        f"[send-back] {ckpt_dir / 'latest.pt'} and {tok_dest}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--output-dir", type=Path, default=HERE / "run-output")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Starting checkpoint (default: {DEFAULT_START_CKPT}). Required unless --from-scratch or --smoke.",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="Random init. Default is to resume a handed-off latest.pt.",
    )
    p.add_argument("--fresh", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if os.environ.get("BABBLE_PRETRAIN_SMOKE") == "1":
        args.smoke = True
    from_scratch = bool(args.from_scratch or args.fresh or args.smoke)
    cfg = RunConfig.from_json(args.config)
    if args.token_budget is not None:
        cfg.token_budget = args.token_budget
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.workers is not None:
        cfg.workers = args.workers
    device = resolve_device(args.device)
    print(f"[device] {device} (cuda available: {torch.cuda.is_available()})", flush=True)
    try:
        train(
            cfg,
            args.output_dir,
            device,
            smoke=args.smoke,
            from_scratch=from_scratch,
            checkpoint=args.checkpoint,
        )
    except CheckpointError as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
