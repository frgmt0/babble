"""Shared fp32 reference for the extreme-inference experiments.

Every alternative runtime (custom torch loop, native C++ engine, llama.cpp)
is judged against the same thing: the live HF path's fp32 logits for a fixed
set of synthetic, role-transcript-formatted prompts. Nothing here reads the
consented corpus -- the texts are written inline.

    python bench/extreme/reference.py build     # writes artifacts/perf-ref/ref.pt
    python bench/extreme/reference.py check MOD:FN
        # FN(list[int]) -> torch.Tensor[T, vocab] full-sequence logits

`compare()` is also importable for use from other harnesses.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "artifacts" / "hf-booper-multiturn-v1"
REF_PATH = ROOT / "artifacts" / "perf-ref" / "ref.pt"

# (prompt transcript, assistant response). Short, medium, and long contexts so
# TTFT-shaped (prefill) paths get exercised at several lengths.
_LONG_HISTORY = "\n".join(
    f"user: {u}\nassistant: {a}"
    for u, a in [
        ("hey booper whats up", "not much just vibing"),
        ("did you see the game last night", "yeah that last goal was insane"),
        ("i think the ref was kinda bad tho", "lol the ref is always bad"),
        ("what should i eat for dinner", "pizza is always the answer"),
        ("i had pizza yesterday", "then tacos, obviously"),
        ("ok tacos it is. do you like cooking", "i burn water tbh"),
        ("lmao same. my roommate cooks everything", "free food is the best food"),
        ("true. hes making curry tonight", "save me some"),
    ]
    * 3
)
CASES: list[tuple[str, str]] = [
    ("user: hey", "hey whats up"),
    ("user: do you like cats or dogs", "cats for sure, they just vibe"),
    (
        "user: can you tell me a short story about a robot who learns to laugh",
        "once there was a robot named bolt who never understood jokes. one day a kid "
        "told him a pun about batteries and something in his circuits just clicked. "
        "he laughed so hard his fans spun up and everyone in the lab laughed with him.",
    ),
    (
        "user: whats your favorite game\nassistant: minecraft probably\n"
        "user: why minecraft",
        "you can build whatever you want and nobody tells you what to do",
    ),
    (_LONG_HISTORY + "\nuser: anyway what are you up to this weekend", "probably just sleeping and playing games lol"),
]


def load_hf():
    from tokenizers import Tokenizer

    from babble.hfserve import _load_int8

    model, config = _load_int8(MODEL_DIR)
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    return model, config, tok


def encode_case(tok, prompt: str, response: str) -> tuple[list[int], int]:
    """`<bos> prompt <sep> response <eos>` ids and the index of `<sep>`."""
    bos, sep, eos = (tok.token_to_id(t) for t in ("<bos>", "<sep>", "<eos>"))
    p = tok.encode(prompt, add_special_tokens=False).ids
    r = tok.encode(response, add_special_tokens=False).ids
    ids = [bos, *p, sep, *r, eos]
    return ids, 1 + len(p)


def build() -> None:
    torch.manual_seed(0)
    model, config, tok = load_hf()
    out = []
    with torch.inference_mode():
        for prompt, response in CASES:
            ids, sep_at = encode_case(tok, prompt, response)
            logits = model(torch.tensor([ids])).logits[0].float()
            out.append({"ids": ids, "sep_at": sep_at, "logits": logits.half()})
            print(f"case len={len(ids)} sep_at={sep_at}", flush=True)
    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_dir": str(MODEL_DIR), "cases": out}, REF_PATH)
    print(f"wrote {REF_PATH}")


def _response_nll(logits: torch.Tensor, ids: list[int], sep_at: int) -> tuple[float, int]:
    """Summed NLL over response+eos tokens (positions after `<sep>`)."""
    targets = torch.tensor(ids[sep_at + 1 :])
    preds = logits[sep_at : len(ids) - 1].float()
    return float(F.cross_entropy(preds, targets, reduction="sum")), len(targets)


def compare(logits_fn) -> dict:
    """Score a candidate `logits_fn(ids) -> [T, vocab]` against the reference.

    Reports max |Δlogit|, top-1 agreement over every position, and response
    NLL/token for reference vs candidate (the number that has to stay put).
    """
    ref = torch.load(REF_PATH)
    max_diff, agree, total = 0.0, 0, 0
    ref_nll = cand_nll = 0.0
    n_tok = 0
    for case in ref["cases"]:
        ids, sep_at = case["ids"], case["sep_at"]
        r = case["logits"].float()
        c = logits_fn(ids).float()
        if c.shape != r.shape:
            raise ValueError(f"shape {tuple(c.shape)} != reference {tuple(r.shape)}")
        max_diff = max(max_diff, float((c - r).abs().max()))
        agree += int((c.argmax(-1) == r.argmax(-1)).sum())
        total += r.shape[0]
        a, n = _response_nll(r, ids, sep_at)
        b, _ = _response_nll(c, ids, sep_at)
        ref_nll += a
        cand_nll += b
        n_tok += n
    return {
        "max_abs_logit_diff": max_diff,
        "top1_agreement": agree / total,
        "ref_nll_per_tok": ref_nll / n_tok,
        "cand_nll_per_tok": cand_nll / n_tok,
        "delta_nll_per_tok": (cand_nll - ref_nll) / n_tok,
        "ref_ppl": math.exp(ref_nll / n_tok),
        "cand_ppl": math.exp(cand_nll / n_tok),
        "positions": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    chk = sub.add_parser("check")
    chk.add_argument("target", help="module:function returning full-sequence logits")
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    else:
        mod, fn = args.target.split(":")
        sys.path.insert(0, str(ROOT))
        print(compare(getattr(importlib.import_module(mod), fn)))


if __name__ == "__main__":
    main()
