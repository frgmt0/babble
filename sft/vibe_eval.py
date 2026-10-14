"""Before/after vibe + coherence eval on booper's real users' prompts.

Samples replies from an INT8 export (the layout `babble.hfserve._load_int8`
reads, e.g. `runs/longctx-v1/export` or `runs/longctx-v2/export`)
to a seeded random subset of the consented snapshot's real direct prompts
(corpus.jsonl rows with source mention/dm/reply), through the same
`sample()` path sft_longform uses for training-time samples, at the serving
temperature. Each reply is scored with `sft.vibe.vibe_score` and the
`coherence_checks` proxy (non-empty, not an echo, not a loop, engages the
prompt).

Only aggregate numbers go to stdout. Per-reply rows (which contain users'
messages) are written to ``--out``, which defaults to the gitignored runs/
directory; never copy that file into the repo.

    .venv/bin/python sft/vibe_eval.py --model runs/longctx-v1/export \\
        --snapshot data/server-snapshot-2026-10-14/live data/server-snapshot-2026-10-14/boopit \\
        --n 200 --seed 0 --out runs/vibe-eval/longctx-v1.json
    .venv/bin/python sft/vibe_eval.py --model runs/longctx-v2/export --snapshot ... --out runs/vibe-eval/longctx-v2.json
    .venv/bin/python sft/vibe_eval.py --compare runs/vibe-eval/longctx-v1.json runs/vibe-eval/longctx-v2.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sft.vibe import coherence_checks, corpus_prompts, vibe_features, vibe_score  # noqa: E402

SERVE_TEMPERATURE = 0.8  # what the live bot samples at (sft_longform.sample's training default is 0.5)
DEFAULT_OUT_DIR = ROOT / "runs" / "vibe-eval"


def pick_prompts(prompts: list[str], n: int, seed: int) -> list[str]:
    """Seeded subset, stable for a given (prompt set, n, seed) so two models see the same prompts."""
    ordered = sorted(prompts)
    rng = random.Random(seed)
    return rng.sample(ordered, min(n, len(ordered)))


def summarize(rows: list[dict], min_vibe: float = 0.75) -> dict:
    """Aggregate vibe + coherence stats over per-reply rows."""
    if not rows:
        raise ValueError("no replies to summarize")
    scores = sorted(r["vibe"] for r in rows)

    def frac(key: str) -> float:
        return round(sum(bool(r["checks"][key]) for r in rows) / len(rows), 4)

    def q(p: float) -> float:
        return round(scores[min(len(scores) - 1, int(p * len(scores)))], 3)

    feature_hits: dict[str, int] = {}
    for r in rows:
        for name in r["features"]:
            feature_hits[name] = feature_hits.get(name, 0) + 1
    return {
        "replies": len(rows),
        "vibe_mean": round(statistics.fmean(scores), 4),
        "vibe_p10": q(0.10),
        "vibe_p50": q(0.50),
        "vibe_p90": q(0.90),
        f"vibe_ge_{min_vibe}": round(sum(s >= min_vibe for s in scores) / len(scores), 4),
        "coherent": frac("coherent"),
        "engaged": frac("engaged"),
        "echo": frac("echo"),
        "degenerate": frac("degenerate"),
        "empty": frac("empty"),
        "assistant_speak": frac("assistant_speak"),
        "mean_reply_tokens": round(statistics.fmean(r["tokens"] for r in rows), 2),
        "feature_rates": {k: round(v / len(rows), 4) for k, v in sorted(feature_hits.items())},
    }


def score_replies(samples: list[dict]) -> list[dict]:
    """`sample()` output rows -> rows with vibe score, features and coherence checks."""
    return [
        {
            **s,
            "vibe": vibe_score(s["reply"]),
            "features": vibe_features(s["reply"]),
            "checks": coherence_checks(s["prompt"], s["reply"]),
        }
        for s in samples
    ]


def run_eval(model_dir: Path, prompts: list[str], *, device: str = "cpu", draws: int = 1, max_new: int = 120,
             temperature: float = SERVE_TEMPERATURE, seed: int = 0, prompt_budget: int | None = None, log=print) -> list[dict]:
    import torch
    from tokenizers import Tokenizer

    from sft.sft_longform import load_base, sample

    torch.manual_seed(seed)
    dev = torch.device(device)
    model, config = load_base(Path(model_dir), dev, log)
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    budget = prompt_budget or int(getattr(config, "babble_prompt_budget", 0) or 512)
    with torch.no_grad():
        samples = sample(model, tok, dev, max_new=max_new, draws=draws, prompt_budget=budget, prompts=prompts, temperature=temperature)
    return score_replies(samples)


def compare(a: dict, b: dict) -> dict:
    """b - a for every shared numeric summary key."""
    sa, sb = a["summary"], b["summary"]
    return {k: round(sb[k] - sa[k], 4) for k in sa if isinstance(sa[k], (int, float)) and isinstance(sb.get(k), (int, float))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="INT8 export dir (model-int8.safetensors + config.json + tokenizer.json)")
    ap.add_argument("--snapshot", nargs="*", type=Path, default=[], help="consented snapshot dir(s) with data/corpus.jsonl")
    ap.add_argument("--n", type=int, default=200, help="prompts to sample (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=SERVE_TEMPERATURE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt-budget", type=int, default=None, help="default: babble_prompt_budget from the export's config")
    ap.add_argument("--min-vibe", type=float, default=0.75)
    ap.add_argument("--out", type=Path, default=None, help="per-reply JSON (contains user prompts; keep under gitignored runs/)")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"), help="diff two --out files and exit")
    args = ap.parse_args(argv)

    if args.compare:
        before, after = (json.loads(p.read_text(encoding="utf-8")) for p in args.compare)
        if before["prompt_sha256"] != after["prompt_sha256"]:
            print("warning: the two evals used different prompt sets", file=sys.stderr)
        print(json.dumps({"before": before["summary"], "after": after["summary"], "delta": compare(before, after)}, indent=2))
        return 0
    if not args.model or not args.snapshot:
        ap.error("--model and --snapshot are required (or use --compare)")
    import hashlib

    prompts = pick_prompts(corpus_prompts(args.snapshot), args.n, args.seed)
    if not prompts:
        ap.error("no usable prompts in the snapshot")
    rows = run_eval(args.model, prompts, device=args.device, draws=args.draws, max_new=args.max_new,
                    temperature=args.temperature, seed=args.seed, prompt_budget=args.prompt_budget,
                    log=lambda m: print(m, file=sys.stderr))
    summary = summarize(rows, args.min_vibe)
    result = {
        "model": str(args.model),
        "seed": args.seed,
        "n_prompts": len(prompts),
        "draws": args.draws,
        "temperature": args.temperature,
        "max_new": args.max_new,
        # identifies the prompt set without storing it in the summary
        "prompt_sha256": hashlib.sha256("\x1f".join(prompts).encode("utf-8")).hexdigest(),
        "summary": summary,
        "rows": rows,
    }
    out = args.out or DEFAULT_OUT_DIR / f"{args.model.name or 'model'}-seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print(f"per-reply rows (contain user prompts, keep private): {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
