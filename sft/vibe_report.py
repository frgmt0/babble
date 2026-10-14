"""Data report for longctx-v2 (unhinged-mac.json): how sft/vibe.py's filter behaves on real data.

Streams N Discord-Dialogues conversations (pinned revision) and reports the
pass rate at the chosen thresholds plus a threshold sweep, score
distributions, rejection reasons, feature hit rates, and the top-scoring and
a seeded sample of rejected conversations (public dataset, printed in full).

With ``--snapshot DIR ...`` it also scores the consented snapshot's real
corpus messages and checks its train.jsonl pairs. For our users' data only
aggregate numbers are printed or saved: no message text ever leaves the
gitignored data/ directory through this script.

    .venv/bin/python sft/vibe_report.py --n 5000 \\
        --snapshot data/server-snapshot-2026-10-14/live data/server-snapshot-2026-10-14/boopit \\
        --json runs/vibe-report/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from babble.blocklist import Blocklist  # noqa: E402
from sft.sft_longform import _chatml_turns, _turn_records, rewrite_gifs  # noqa: E402
from sft.vibe import (  # noqa: E402
    DEFAULT_MIN_TARGETS,
    DEFAULT_MIN_VIBE,
    LocalStats,
    conversation_verdict,
    corpus_prompts,
    local_snapshot_records,
    snapshot_corpus_file,
    trim_to_user_start,
    vibe_features,
    vibe_score,
)

DEFAULT_PRESET = ROOT / "configs" / "sft" / "longctx-mac.json"
SWEEP_VIBE = (0.0, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25)
SWEEP_TARGETS = (1, 2, 3)


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def q(p: float) -> float:
        return round(ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))], 3)

    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
    }


def feature_rates(texts: list[str]) -> dict[str, float]:
    hits: Counter = Counter()
    for text in texts:
        hits.update(vibe_features(text).keys())
    return {name: round(n / max(len(texts), 1), 4) for name, n in sorted(hits.items())}


def histogram(values: list[float], edges=(-3, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 4)) -> dict[str, int]:
    out: dict[str, int] = {}
    labels = [f"<{edges[0]}"] + [f"[{a},{b})" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}"]
    for label in labels:
        out[label] = 0
    for v in values:
        if v < edges[0]:
            out[labels[0]] += 1
        elif v >= edges[-1]:
            out[labels[-1]] += 1
        else:
            for i, (a, b) in enumerate(zip(edges, edges[1:])):
                if a <= v < b:
                    out[labels[i + 1]] += 1
                    break
    return out


def stream_rows(n: int, revision: str | None, skip_rows: int = 0):
    from datasets import load_dataset

    ds = load_dataset("mookiezi/Discord-Dialogues", split="train", streaming=True, revision=revision)
    if skip_rows:
        ds = ds.skip(skip_rows)
    for i, row in enumerate(ds):
        if i >= n:
            break
        yield row["text"]


def discord_report(raws: list[str], *, min_vibe: float, min_targets: int, blocklist: Blocklist, show: int, seed: int) -> dict:
    reasons: Counter = Counter()
    target_scores: list[float] = []
    user_scores: list[float] = []
    group_means: list[float] = []
    targets_text: list[str] = []
    passed: list[tuple[float, list[tuple[str, str]]]] = []
    rejected: list[tuple[str, list[tuple[str, str]]]] = []
    sweep = {f"vibe>={v}|targets>={t}": 0 for v in SWEEP_VIBE for t in SWEEP_TARGETS}
    records_per_row: list[int] = []
    kept_targets = 0
    for raw in raws:
        turns = _chatml_turns(raw)
        # `discord` (unfiltered) drops rows that open with the assistant speaker.
        records_per_row.append(len(_turn_records("discord", "x", turns)))
        judged = [(role, rewrite_gifs(text)[0]) for role, text in trim_to_user_start(turns)]
        for role, text in judged:
            (target_scores if role == "assistant" else user_scores).append(vibe_score(text))
            if role == "assistant":
                targets_text.append(text)
        verdict = conversation_verdict(judged, min_vibe=min_vibe, min_targets=min_targets, blocklist=blocklist)
        reasons[verdict.reason] += 1
        if verdict.scores:
            group_means.append(verdict.mean_vibe)
        if verdict.ok:
            passed.append((verdict.mean_vibe, judged))
            kept_targets += verdict.targets
        else:
            rejected.append((verdict.reason, judged))
        for v in SWEEP_VIBE:
            for t in SWEEP_TARGETS:
                if conversation_verdict(judged, min_vibe=v, min_targets=t, blocklist=blocklist).ok:
                    sweep[f"vibe>={v}|targets>={t}"] += 1
    n = len(raws)
    rows_with_targets = sum(1 for r in records_per_row if r)
    rng = random.Random(seed)
    passed.sort(key=lambda item: -item[0])
    rejected_sample = rng.sample(rejected, min(show, len(rejected)))
    return {
        "rows": n,
        "thresholds": {"min_vibe": min_vibe, "min_targets": min_targets},
        "passed": reasons["ok"],
        "pass_rate": round(reasons["ok"] / max(n, 1), 4),
        "kept_targets": kept_targets,
        "kept_targets_per_row": round(kept_targets / max(n, 1), 4),
        "reasons": dict(reasons.most_common()),
        "sweep_pass_rate": {k: round(v / max(n, 1), 4) for k, v in sweep.items()},
        "discord_records_per_row": round(sum(records_per_row) / max(n, 1), 4),
        "rows_with_targets_frac": round(rows_with_targets / max(n, 1), 4),
        "assistant_target_scores": percentiles(target_scores),
        "assistant_target_hist": histogram(target_scores),
        "user_turn_scores": percentiles(user_scores),
        "group_mean_scores": percentiles(group_means),
        "assistant_feature_rates": feature_rates(targets_text),
        "top": [{"mean_vibe": round(m, 3), "turns": t} for m, t in passed[:show]],
        "rejected": [{"reason": r, "turns": t} for r, t in rejected_sample],
    }


def supply_scan(rows, *, base_targets: int, min_vibe: float, min_targets: int, blocklist: Blocklist, gif_tags: bool = True) -> dict:
    """Walk the stream from row 0: where did the base's `discord` slice end, and
    how much discord_vibe data lies before / after that point?

    Discord-Dialogues is ordered by conversation length (longest first), so
    multi-target conversations exist only near the head; this measures the
    real supply instead of extrapolating from a sample.
    """
    import time

    from sft.sft_longform import GifStats

    t0 = time.perf_counter()
    cumulative = 0
    base_conversations = 0
    base_end_row = None
    before = {"groups": 0, "targets": 0}
    after = {"groups": 0, "targets": 0}
    multi_target_rows_after = 0
    scanned = 0
    last_multi_row = None
    for i, raw in enumerate(rows):
        scanned += 1
        turns = _chatml_turns(raw)
        n = len(_turn_records("discord", "x", turns, GifStats() if gif_tags else None))
        if base_end_row is None and n:
            base_conversations += 1
        cumulative += n
        if base_end_row is None and cumulative >= base_targets:
            base_end_row = i + 1
        judged = [(role, rewrite_gifs(text)[0]) for role, text in trim_to_user_start(turns)]
        pairs = len(judged) // 2
        if pairs >= min_targets:
            last_multi_row = i + 1
            if base_end_row is not None and i + 1 > base_end_row:
                multi_target_rows_after += 1
        verdict = conversation_verdict(judged, min_vibe=min_vibe, min_targets=min_targets, blocklist=blocklist)
        if verdict.ok:
            bucket = after if base_end_row is not None and i + 1 > base_end_row else before
            bucket["groups"] += 1
            bucket["targets"] += verdict.targets
    elapsed = time.perf_counter() - t0
    return {
        "rows_scanned": scanned,
        "seconds": round(elapsed, 1),
        "rows_per_s": round(scanned / max(elapsed, 1e-9), 1),
        "base_conversations_used": base_conversations,
        "base_targets": base_targets,
        "base_end_row": base_end_row,
        "last_row_with_min_targets": last_multi_row,
        "multi_target_rows_after_base": multi_target_rows_after,
        "vibe_before_base_end": before,
        "vibe_after_base_end": after,
    }


def snapshot_report(paths: list[Path], blocklist: Blocklist) -> dict:
    """Aggregate-only view of the consented snapshot (no message text)."""
    out: dict = {"corpus": {}, "local_pairs": {}}
    by_source: dict[str, list[float]] = defaultdict(list)
    texts_by_source: dict[str, list[str]] = defaultdict(list)
    blocked = 0
    for path in paths:
        try:
            file = snapshot_corpus_file(path)
        except FileNotFoundError:
            out["corpus"][str(path)] = "missing corpus.jsonl"
            continue
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            if blocklist.matches(text):
                blocked += 1
                continue
            by_source[row.get("source", "?")].append(vibe_score(text))
            texts_by_source[row.get("source", "?")].append(text)
    direct = [s for src in ("mention", "dm", "reply") for s in by_source.get(src, [])]
    direct_texts = [t for src in ("mention", "dm", "reply") for t in texts_by_source.get(src, [])]
    out["corpus"] = {
        "blocklisted_skipped": blocked,
        "by_source": {src: percentiles(v) for src, v in sorted(by_source.items())},
        "direct_prompts": percentiles(direct),
        "direct_prompts_hist": histogram(direct),
        "direct_feature_rates": feature_rates(direct_texts),
        "distinct_eval_prompts": len(corpus_prompts(paths, blocklist=blocklist)) if direct else 0,
    }
    for path in paths:
        stats = LocalStats()
        try:
            local_snapshot_records([path], blocklist=blocklist, stats=stats)
            out["local_pairs"][str(path)] = stats.as_dict()
        except FileNotFoundError as e:
            out["local_pairs"][str(path)] = {"error": str(e)}
        except ValueError:
            out["local_pairs"][str(path)] = {**stats.as_dict(), "error": "no usable rows"}
    return out


MIX_KEYS = (
    "mix_story", "mix_wp", "mix_norobots", "mix_smoltalk", "mix_discord", "mix_ultrachat", "mix_dolly", "mix_oasst",
    "mix_smol_short", "mix_nq", "mix_arith", "mix_persona", "mix_wiki", "mix_discord_vibe", "mix_local",
)
# sft_longform's argparse defaults for the three legacy mixes a preset may omit.
MIX_DEFAULTS = {"mix_story": 0.45, "mix_norobots": 0.15, "mix_discord": 0.40}


def consumed_discord_targets(preset: dict) -> dict:
    """How many `discord` targets a preset's build took from the head of the stream.

    build_examples collects ``int(examples * w / total)`` train targets plus
    ``round(val_examples * w / total)`` val targets (repeat is 1 for discord).
    The row where that count is reached is measured by ``supply_scan``: the
    dataset is ordered by length, so a per-row average from one sample would
    be wrong.
    """
    weights = {k: float(preset.get(k, MIX_DEFAULTS.get(k, 0.0))) for k in MIX_KEYS}
    total = sum(weights.values())
    w = weights["mix_discord"]
    want = int(preset["examples"] * w / total)
    want_val = max(1, round(preset.get("val_examples", 400) * w / total))
    targets = want + want_val
    return {"targets": targets}


def print_report(report: dict) -> None:
    d = report["discord"]
    print(f"== Discord-Dialogues @ {report['revision']} (rows {d['rows']}, skip {report['skip_rows']}) ==")
    print(f"thresholds {d['thresholds']}: passed {d['passed']} ({d['pass_rate']:.1%}), kept targets {d['kept_targets']} "
          f"({d['kept_targets_per_row']:.3f}/row)")
    print(f"reasons: {d['reasons']}")
    print(f"discord records/row (unfiltered `discord` source): {d['discord_records_per_row']}  rows with targets: {d['rows_with_targets_frac']:.1%}")
    print(f"base preset `discord` consumption ({report['base_preset']}): {report['base_consumed']}")
    print("sweep (pass rate):")
    for t in SWEEP_TARGETS:
        print("  targets>=%d  " % t + "  ".join(f"vibe>={v}: {d['sweep_pass_rate'][f'vibe>={v}|targets>={t}']:.1%}" for v in SWEEP_VIBE))
    print(f"assistant target scores: {d['assistant_target_scores']}")
    print(f"assistant target hist: {d['assistant_target_hist']}")
    print(f"user turn scores: {d['user_turn_scores']}")
    print(f"group mean scores: {d['group_mean_scores']}")
    print(f"assistant feature hit rates: {d['assistant_feature_rates']}")
    print(f"\n-- top {len(d['top'])} passing conversations --")
    for item in d["top"]:
        print(f"[mean {item['mean_vibe']}]")
        for role, text in item["turns"][:8]:
            print(f"   {role[:1]}: {text[:140]!r}")
    print(f"\n-- {len(d['rejected'])} rejected (seeded sample) --")
    for item in d["rejected"]:
        print(f"[{item['reason']}]")
        for role, text in item["turns"][:6]:
            print(f"   {role[:1]}: {text[:140]!r}")
    if "supply" in report:
        print(f"\n== supply scan ==\n{report['supply']}")
    if "snapshot" in report:
        s = report["snapshot"]
        print("\n== consented snapshot (aggregate only) ==")
        print(f"corpus by source: {s['corpus'].get('by_source')}")
        print(f"direct prompts (mention/dm/reply): {s['corpus'].get('direct_prompts')}")
        print(f"direct prompts hist: {s['corpus'].get('direct_prompts_hist')}")
        print(f"direct feature hit rates: {s['corpus'].get('direct_feature_rates')}")
        print(f"blocklisted skipped: {s['corpus'].get('blocklisted_skipped')}  distinct eval prompts: {s['corpus'].get('distinct_eval_prompts')}")
        print(f"local pairs: {s['local_pairs']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000, help="Discord-Dialogues rows to stream")
    ap.add_argument("--revision", default=None, help="dataset revision (default: discord_revision from --preset)")
    ap.add_argument("--preset", type=Path, default=DEFAULT_PRESET, help="base model's preset: pinned discord_revision + how many rows its `discord` slice consumed")
    ap.add_argument("--skip-rows", type=int, default=0, help="start this many rows into the stream")
    ap.add_argument("--min-vibe", type=float, default=DEFAULT_MIN_VIBE, help="minimum mean target vibe for the headline pass rate")
    ap.add_argument("--min-targets", type=int, default=DEFAULT_MIN_TARGETS)
    ap.add_argument("--snapshot", nargs="*", default=[], type=Path, help="consented snapshot dir(s) (aggregate stats only)")
    ap.add_argument("--supply-rows", type=int, default=0, help="also scan this many rows from the head: where the base's discord slice ended and the discord_vibe supply before/after it")
    ap.add_argument("--show", type=int, default=15, help="top / rejected conversations to print")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="also write the full report here (aggregate-only for the snapshot)")
    args = ap.parse_args(argv)
    revision = args.revision or json.loads(args.preset.read_text(encoding="utf-8"))["discord_revision"]
    blocklist = Blocklist.load()
    raws = list(stream_rows(args.n, revision, args.skip_rows))
    discord = discord_report(raws, min_vibe=args.min_vibe, min_targets=args.min_targets, blocklist=blocklist, show=args.show, seed=args.seed)
    report = {
        "revision": revision,
        "skip_rows": args.skip_rows,
        "blocklist_terms": len(blocklist.terms),
        "base_preset": str(args.preset.relative_to(ROOT) if args.preset.is_absolute() and ROOT in args.preset.parents else args.preset),
        "base_consumed": consumed_discord_targets(json.loads(args.preset.read_text(encoding="utf-8"))),
        "discord": discord,
    }
    if args.supply_rows:
        report["supply"] = supply_scan(
            stream_rows(args.supply_rows, revision, 0),
            base_targets=report["base_consumed"]["targets"],
            min_vibe=args.min_vibe,
            min_targets=args.min_targets,
            blocklist=blocklist,
        )
    if args.snapshot:
        report["snapshot"] = snapshot_report(args.snapshot, blocklist)
    print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    code = main()
    sys.stdout.flush()
    # The streaming HTTP client can leave a retrying worker behind after the
    # last row; exit without waiting on it.
    import os

    os._exit(code)
