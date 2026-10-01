"""Held-out QA eval for booper: run a local HF export through babble's own
serving path and score the replies.

    python bench/qa/run_eval.py --model-dir DIR [--label NAME] [--out results.json] [--limit N]

What "production path" means here:

* The generator is `babble.hfserve.make_generator(settings)` with
  ``serve_backend=hf`` and ``hf_runtime`` (default ``native``, the live
  runtime; it falls back to lean/transformers exactly like the bot does).
* The prompt is built the way `babble.core.Babble._generation_prompt` builds
  it: if the export's ``config.json`` says
  ``babble_prompt_format: role_transcript_v1``, conversation context is on and
  the backend's own ``conversation_prompt`` serializes ``history`` + question
  as a ``user: / assistant:`` transcript under the live budgets (turns =
  ``babble_history_turns``, 512 prompt tokens, no char cap). Otherwise the
  model gets the bare question (the single-turn behavior; ``history`` is then
  dropped, as the bot would drop it -- such models cannot pass ``followup``).
  The backend wraps it as ``<bos> PROMPT <sep>`` and decodes to ``<eos>``.

Decoding. Sampler settings are the live ones (Settings defaults + the live
.env): temperature 0.5, top_k 40, top_p 0.9, repetition_penalty 1.15,
no_repeat_ngram_size 4, best_of 4, frequency/presence penalties off (gated off
in the HF runtimes), but max_new_tokens is 64.

* default ``--greedy``: top_k=1 and best_of=1 through the same sampler, i.e.
  argmax after the production repetition penalty / n-gram ban. Deterministic.
* ``--sampled``: the full live sampler, with ``torch.manual_seed(seed + i)``
  before item ``i`` (the native/lean runtimes draw their RNG seed from torch),
  so a rerun reproduces the same replies on the same build.

Outputs: ``--out`` JSON (config, aggregate, every item + response + score) and
a markdown summary next to it (``.md``), plus a summary on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.qa.scoring import FLAGS, aggregate, score_item  # noqa: E402

CATEGORY_ORDER = ("fact_common", "math", "definition", "about_bot", "commonsense", "followup", "chat")

LIVE_SAMPLING = dict(
    temperature=0.5,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.15,
    no_repeat_ngram_size=4,
    best_of=4,
)


def load_questions(path: Path, limit: int | None = None, categories: set[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if categories:
        rows = [r for r in rows if r["category"] in categories]
    if limit:
        rows = rows[:limit]
    return rows


def history_turns(history: list[dict] | None):
    """[{role, content}, ...] -> [ConversationTurn] (user/assistant pairs)."""
    from babble.conversation import ConversationTurn

    turns, pending = [], None
    for msg in history or []:
        role, content = msg.get("role"), str(msg.get("content", ""))
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            turns.append(ConversationTurn(user=pending, assistant=content))
            pending = None
    return turns


def build_settings(model_dir: Path, *, runtime: str, threads: int, greedy: bool,
                   max_new_tokens: int, max_prompt_tokens: int, cfg: dict):
    from babble.config import Settings

    scratch = Path(tempfile.mkdtemp(prefix="babble-qa-"))
    s = Settings.for_root(scratch)  # never touches a real data dir
    s.serve_backend = "hf"
    s.hf_runtime = runtime
    s.hf_model_dir = model_dir
    s.infer_threads = threads
    s.train_threads = threads
    for k, v in LIVE_SAMPLING.items():
        setattr(s, k, v)
    if greedy:
        s.top_k = 1
        s.best_of = 1
    s.max_new_tokens = max_new_tokens
    s.lean_prefix_cache_mb = 0  # every item is an independent cold prompt
    s.lean_prefix_cache_entries = 0
    s.conversation_context = cfg.get("babble_prompt_format") == "role_transcript_v1"
    s.conversation_max_turns = int(cfg.get("babble_history_turns", 6) or 6)
    s.conversation_max_tokens = max_prompt_tokens
    s.conversation_max_chars = 0
    return s


def make_prompt(gen, settings, item: dict) -> str:
    """Exactly what `Babble._generation_prompt` would hand the generator."""
    from babble.conversation import conversation_prompt

    q = item["question"]
    if not settings.conversation_context:
        return q
    hist = tuple(history_turns(item.get("history")))
    kwargs = dict(
        max_turns=settings.conversation_max_turns,
        max_tokens=settings.conversation_max_tokens,
        max_chars=settings.conversation_max_chars,
    )
    fmt = getattr(gen, "conversation_prompt", None)
    if callable(fmt):
        return str(fmt(hist, q, **kwargs))
    return conversation_prompt(hist, q, max_turns=kwargs["max_turns"], max_chars=kwargs["max_chars"])


def fmt_pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def markdown_summary(result: dict) -> str:
    agg, cfg = result["aggregate"], result["config"]
    lines = [
        f"# QA eval: {cfg['label']}",
        "",
        f"- model dir: `{cfg['model_dir']}`",
        f"- runtime: {cfg['runtime_used']}, decoding: {cfg['decoding']}, max_new_tokens {cfg['max_new_tokens']}",
        f"- prompt format: {cfg['prompt_format']}, items: {agg['n']}, wall: {cfg['wall_s']:.0f}s",
        "",
        f"**overall accuracy {fmt_pct(agg['overall_accuracy'])}%** (micro, answerable items) | "
        f"macro {fmt_pct(agg['macro_accuracy'])}% | clean (no flag) {fmt_pct(agg['clean_rate'])}%",
        "",
        "| category | n | acc % | echo % | empty % | non_answer % | repetitive % | clean % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    cats = agg["categories"]
    for name in [c for c in CATEGORY_ORDER if c in cats] + [c for c in cats if c not in CATEGORY_ORDER]:
        c = cats[name]
        lines.append(
            f"| {name} | {c['n']} | {fmt_pct(c['accuracy'])} | "
            + " | ".join(fmt_pct(c[f"{f}_rate"]) for f in FLAGS)
            + f" | {fmt_pct(c['clean_rate'])} |"
        )
    lines.append(
        f"| **all** | {agg['n']} | {fmt_pct(agg['overall_accuracy'])} | "
        + " | ".join(fmt_pct(agg[f"{f}_rate"]) for f in FLAGS)
        + f" | {fmt_pct(agg['clean_rate'])} |"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, default=None, help="results JSON (default bench/qa/results/<label>.json)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--categories", default=None, help="comma-separated subset")
    ap.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--greedy", dest="greedy", action="store_true", default=True, help="(default) argmax decoding")
    mode.add_argument("--sampled", dest="greedy", action="store_false", help="live sampler with a fixed seed")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--runtime", default="native", choices=("native", "lean", "transformers"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-prompt-tokens", type=int, default=512)
    args = ap.parse_args(argv)

    model_dir = args.model_dir.expanduser().resolve()
    label = args.label or model_dir.name
    out = args.out or (HERE / "results" / f"{label}.json")
    cfg = json.loads((model_dir / "config.json").read_text())

    # The eval must not inherit a live box's serving knobs from the shell.
    for var in ("BABBLE_NATIVE_SPEC", "BABBLE_HF_FREQUENCY_PENALTIES"):
        os.environ.pop(var, None)

    import torch

    from babble.hfserve import make_generator

    settings = build_settings(
        model_dir, runtime=args.runtime, threads=args.threads, greedy=args.greedy,
        max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens, cfg=cfg,
    )
    cats = set(args.categories.split(",")) if args.categories else None
    items = load_questions(args.questions, args.limit, cats)

    t0 = time.perf_counter()
    gen = make_generator(settings)
    load_s = time.perf_counter() - t0
    runtime_used = getattr(gen, "runtime", None) or type(gen).__name__
    print(f"loaded {model_dir.name} via {runtime_used} in {load_s:.1f}s; {len(items)} items", file=sys.stderr)

    rows = []
    t1 = time.perf_counter()
    for i, item in enumerate(items):
        prompt = make_prompt(gen, settings, item)
        if not args.greedy:
            torch.manual_seed(args.seed + i)
        started = time.perf_counter()
        response = gen(prompt).text
        ms = (time.perf_counter() - started) * 1000
        sc = score_item(item, response)
        rows.append({**item, "prompt": prompt, "response": response, "ms": round(ms, 1), "score": sc.to_dict()})
        mark = {True: "ok ", False: "XX ", None: "-- "}[sc.correct]
        print(f"[{i + 1:3d}/{len(items)}] {mark}{item['id']:<16} {response[:90]!r} {','.join(sc.flags)}", file=sys.stderr)
    wall = time.perf_counter() - t1

    result = {
        "config": {
            "label": label,
            "model_dir": str(model_dir),
            "runtime_used": runtime_used,
            "decoding": "greedy (top_k=1, best_of=1)" if args.greedy else f"sampled (seed {args.seed}+i)",
            "sampling": {k: getattr(settings, k) for k in (*LIVE_SAMPLING, "max_new_tokens")},
            "max_new_tokens": settings.max_new_tokens,
            "prompt_format": cfg.get("babble_prompt_format", "raw (single-turn)"),
            "conversation_context": settings.conversation_context,
            "conversation_max_turns": settings.conversation_max_turns,
            "conversation_max_tokens": settings.conversation_max_tokens,
            "questions": str(args.questions),
            "load_s": round(load_s, 2),
            "wall_s": round(wall, 2),
        },
        "aggregate": aggregate(rows),
        "items": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    md = markdown_summary(result)
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out} and {out.with_suffix('.md')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
