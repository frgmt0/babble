"""Samples + bits-per-char comparison after SSH-checkpoint post-train.

Never points at /home/beckett/babble-live. Reads the copied corpus, the
HF tokenizer, the pretrained checkpoint, the post-train candidate, and a
copy of the currently-served live weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from babble.config import Settings
from babble.cpu_runtime import force_cpu_device
from babble.generate import continue_text, load_model
from babble.identity import Pseudonymiser
from babble.model import Babbler, ModelConfig, sequence_loss
from babble.posttrain import eval_examples_with_model
from babble.subword import BPETokenizer
from babble.subword import stack_examples as bpe_stack
from babble.subword import text_examples as bpe_text
from babble.trainer import corpus_rows, eval_loss, split_rows, to_examples

PROBES = ["hola", "hello", "do you want to enter giveway", "why is", "the cat", "where"]
SAMPLES_PER_PROMPT = 3
MAX_NEW = 60


def bits_per_char(nats_total: float, chars: int) -> float:
    return (nats_total / max(chars, 1)) / math.log(2)


@torch.inference_mode()
def sample_bpe(model: Babbler, tok: BPETokenizer, prompt: str, generator: torch.Generator, max_new: int = MAX_NEW) -> str:
    device = next(model.parameters()).device
    ids = [tok.specials.bos, *tok.encode(prompt)]
    ids = ids[-model.config.block_size :]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    cache = model.new_cache(batch_size=1, max_len=min(len(ids) + max_new, model.config.block_size))
    logits = model(idx, cache=cache)
    out_ids = list(ids)
    for _ in range(max_new):
        last = logits[:, -1, :] / 0.5
        k = min(40, last.size(-1))
        v, _ = torch.topk(last, k)
        last = last.masked_fill(last < v[:, [-1]], float("-inf"))
        last[:, [tok.specials.pad, tok.specials.bos, tok.specials.sep]] = float("-inf")
        probs = F.softmax(last, dim=-1)
        next_id = int(torch.multinomial(probs, 1, generator=generator))
        if next_id == tok.specials.eos or cache.length >= cache.max_len:
            break
        out_ids.append(next_id)
        logits = model(torch.tensor([[next_id]], dtype=torch.long, device=device), cache=cache)
    return tok.decode(out_ids[1:])


def load_ckpt(path: Path) -> Babbler:
    device = force_cpu_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def loss_stats_bpe(model: Babbler, examples, pad_id: int) -> tuple[float, int]:
    tokens, mask, weights = bpe_stack(examples, pad_id)
    with torch.inference_mode():
        per = sequence_loss(model, tokens, mask, weights)
        n = int((mask[:, 1:] * weights[:, None]).sum().item())
        return float(per) * n, n


def probes_bpe(model: Babbler, tok: BPETokenizer) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for prompt in PROBES:
        seed_base = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16) % (2**31)
        samples = []
        for i in range(SAMPLES_PER_PROMPT):
            gen = torch.Generator().manual_seed(seed_base + i)
            samples.append(sample_bpe(model, tok, prompt, gen))
        out[prompt] = samples
    return out


def probes_byte(model) -> dict[str, list[str]]:
    settings = Settings.from_env()
    out: dict[str, list[str]] = {}
    for prompt in PROBES:
        seed_base = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16) % (2**31)
        samples = []
        for i in range(SAMPLES_PER_PROMPT):
            gen = torch.Generator().manual_seed(seed_base + i)
            samples.append(
                continue_text(
                    model, prompt, max_new_tokens=MAX_NEW, temperature=0.5, top_k=40, generator=gen
                )
            )
        out[prompt] = samples
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.from_env()
    ids = Pseudonymiser.load(settings)
    tok = BPETokenizer.from_json(root / "artifacts/hf-booper-pretrain/tokenizer.json")
    pre_path = root / "artifacts/hf-booper-pretrain/latest.pt"
    cand_path = settings.checkpoint_dir / "post_candidate.pt"
    live_path = root / "experiments/results/ssh-posttrain/live_served.pt"

    rows = corpus_rows(settings, ids)
    split = split_rows(rows, settings)
    val_chars = sum(len(r.text) for r in split.val)
    val_bpe = [ex for row in split.val for ex in bpe_text(tok, row.text, 1024)]
    val_byte = to_examples(split.val, 512)

    pre = load_ckpt(pre_path)
    nats_pre, ntok_pre = loss_stats_bpe(pre, val_bpe, tok.specials.pad)
    mean_pre = nats_pre / max(ntok_pre, 1)
    probes_pre = probes_bpe(pre, tok)
    del pre

    post = load_ckpt(cand_path)
    nats_post, ntok_post = loss_stats_bpe(post, val_bpe, tok.specials.pad)
    mean_post = nats_post / max(ntok_post, 1)
    probes_post = probes_bpe(post, tok)
    del post

    live_payload = torch.load(live_path, map_location="cpu", weights_only=True)
    live = Babbler(ModelConfig.from_dict(live_payload["config"])).to(force_cpu_device())
    live.load_state_dict(live_payload["model"])
    live.eval()
    live_mean = eval_loss(live, val_byte)
    # total nats: mean * token count
    tokens_t, mask, weights = __import__("babble.trainer", fromlist=["_stack_examples"])._stack_examples(val_byte)
    ntok_live = int((mask[:, 1:] * weights[:, None]).sum().item())
    nats_live = float(live_mean) * ntok_live
    probes_live = probes_byte(live)
    del live

    report = {
        "val_rows": len(split.val),
        "val_chars": val_chars,
        "pretrain": {
            "mean_nats_per_token": mean_pre,
            "tokens": ntok_pre,
            "nats_total": nats_pre,
            "bits_per_char": bits_per_char(nats_pre, val_chars),
            "probes": probes_pre,
        },
        "posttrain": {
            "mean_nats_per_token": mean_post,
            "tokens": ntok_post,
            "nats_total": nats_post,
            "bits_per_char": bits_per_char(nats_post, val_chars),
            "probes": probes_post,
        },
        "live_served": {
            "mean_nats_per_token": live_mean,
            "tokens": ntok_live,
            "nats_total": nats_live,
            "bits_per_char": bits_per_char(nats_live, val_chars),
            "config": live_payload["config"],
            "step": live_payload.get("step"),
            "probes": probes_live,
        },
    }
    out = root / "experiments/results/ssh-posttrain/measure.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "probes"} if isinstance(v, dict) else v
                      for k, v in report.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
