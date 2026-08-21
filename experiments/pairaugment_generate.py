"""Generate the real (LLM-paraphrased) augmented-pair pool the measurement
grid trains on, plus the two automated checks that must run before anything
downstream trusts the file: the leakage check and the register comparison.

Seeds `experiments/pairaugment-scratch/data/` with the 50 stand-in
correction pairs (`pairaugment_data.PAIRS`), generates up to `--n` variants
per TRAIN-side pair using the real `claude` CLI, and writes:

- `experiments/pairaugment-scratch/data/augmented_pairs.jsonl` -- the real
  generated pairs (gitignored: it lives under a `data/` directory, same rule
  as every other data dir in this repo).
- `experiments/results/pairaugment/generate_summary.json` -- counts,
  leakage report, register report (numbers only -- tracked).
- `experiments/results/pairaugment/sample_variants.json` -- a handful of
  full (source, variant) pairs for the report to quote verbatim (these are
  MY fabricated stand-in pairs, not real user data, so there is no consent
  concern in committing a few as illustration).

Usage:
    python -m experiments.pairaugment_generate --n 5 --workers 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

from babble.config import Settings
from babble.consent import ConsentStore
from babble.identity import Pseudonymiser
from babble.pairaugment import (
    AugmentedPairStore,
    assert_no_leakage,
    generate_augmented_pairs,
    register_comparison,
)
from babble.pairsplit import pair_split
from babble.post_state import trainable_pairs
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id

from .pairaugment_data import PAIRS

SCRATCH = Path("experiments/pairaugment-scratch")
RESULTS = Path("experiments/results/pairaugment")


def seed(settings: Settings) -> None:
    ids = Pseudonymiser.load(settings)
    consent = ConsentStore(settings.consent_path)
    consent.grant("ro-stand-in")
    consent.grant("booper-stand-in")
    asker, helper = ids.user("ro-stand-in"), ids.user("booper-stand-in")
    store = InteractionStore(settings.interactions_path)
    for prompt, chosen in PAIRS:
        store.append(
            Interaction(
                id=make_row_id(CORRECTION, prompt, chosen, asker, helper),
                signal=CORRECTION,
                prompt=prompt,
                rejected="(stand-in rejected answer)",
                chosen=chosen,
                prompt_author=asker,
                signal_author=helper,
                weight=settings.correction_weight,
            )
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="max variants per pair (the N sweep subsamples this)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="haiku")
    args = ap.parse_args()

    settings = Settings.for_root(SCRATCH)
    settings.paraphrase_model = args.model
    settings.ensure_dirs()
    if settings.interactions_path.exists():
        settings.interactions_path.unlink()
    seed(settings)

    ids = Pseudonymiser.load(settings)
    pairs = trainable_pairs(settings, ids)
    train_pairs, val_pairs = pair_split(pairs)
    print(f"seeded {len(pairs)} pairs -> {len(train_pairs)} train-side / {len(val_pairs)} val-side")

    result = generate_augmented_pairs(settings, n=args.n, max_workers=args.workers, ids=ids)
    print(
        f"generated {result.generated} variant(s) from {result.train_side_pairs} train-side pair(s) "
        f"({result.val_side_pairs} val-side never touched), "
        f"{result.failed_pairs} pair(s) failed, {result.skipped_blocklist} blocked"
    )
    if result.failures:
        print("failures:", result.failures[:10])

    leak_report = assert_no_leakage(settings, ids)  # raises loudly if this is not clean
    print(
        f"leakage check: {leak_report.checked} checked, {leak_report.train_side} train-side, "
        f"{leak_report.leaked} leaked, {leak_report.orphaned} orphaned -- CLEAN"
    )

    reg_report = register_comparison(settings, ids)
    print(
        f"register: real mean {reg_report.real_mean_chars:.1f} chars / "
        f"{reg_report.real_mean_words:.1f} words, lowercase {reg_report.real_lowercase_rate:.3f} | "
        f"variant mean {reg_report.variant_mean_chars:.1f} chars / "
        f"{reg_report.variant_mean_words:.1f} words, lowercase {reg_report.variant_lowercase_rate:.3f} | "
        f"vocab overlap {reg_report.vocab_overlap:.3f} | drifted={reg_report.drifted}"
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "seeded_pairs": len(pairs),
        "train_side_pairs": len(train_pairs),
        "val_side_pairs": len(val_pairs),
        "generate": dataclasses.asdict(result),
        "leakage": dataclasses.asdict(leak_report),
        "register": dataclasses.asdict(reg_report),
    }
    (RESULTS / "generate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    rng = random.Random(7)
    all_variants = AugmentedPairStore(settings.augmented_pairs_path).all()
    sample = rng.sample(all_variants, k=min(10, len(all_variants)))
    by_source = {p.id: p for p in pairs}
    sample_out = [
        {
            "source_prompt": by_source[v.source_pair_id].prompt if v.source_pair_id in by_source else None,
            "source_chosen": by_source[v.source_pair_id].chosen if v.source_pair_id in by_source else None,
            "variant_index": v.variant_index,
            "variant_prompt": v.prompt,
            "variant_chosen": v.chosen,
        }
        for v in sample
    ]
    (RESULTS / "sample_variants.json").write_text(json.dumps(sample_out, indent=2, ensure_ascii=False))
    print(f"wrote {RESULTS / 'generate_summary.json'} and sample_variants.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
