"""The with/without measurement, the shuffled-word-order control, and the N
sweep -- the "half the job" the project report is built from.

Reads the real augmented pairs `pairaugment_generate.py` wrote to
`experiments/pairaugment-scratch/data/augmented_pairs.jsonl` (train-side
only, already leak-checked), builds a shuffled-word-order control from them,
and runs `post_train` across:

    baseline (0x augmentation), 1x, 3x, 5x, shuffled-3x

each for 3 seeds, isolated in its own scratch checkpoint dir copied from a
shared pretrained snapshot. No corpus rows are seeded, so `post_train`'s
corpus-val gate is inactive (`corpus_val_examples` is empty) and checkpoint
selection falls back to pair val -- the held-out REAL correction-pair loss,
which is exactly the number this feature is supposed to move.

Usage:
    python -m experiments.pairaugment_grid
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

from babble.config import Settings
from babble.consent import ConsentStore
from babble.cpu_runtime import force_cpu_device
from babble.identity import Pseudonymiser
from babble.pairaugment import AugmentedPair, AugmentedPairStore
from babble.posttrain import post_train
from babble.store import InteractionStore

from .pairaugment_data import PAIRS
from .pairaugment_generate import seed as seed_real_pairs

SCRATCH = Path("experiments/pairaugment-scratch")
REAL_DATA = SCRATCH / "data"
GRID_ROOT = SCRATCH / "grid"
RESULTS = Path("experiments/results/pairaugment")

SEEDS = (1, 2, 3)
STEPS = 150
CHECKPOINT_EVERY = 10


def _tiny_model_settings(root: Path) -> Settings:
    s = Settings.for_root(root)
    # A tiny model so the grid runs in seconds, not minutes -- same shrink
    # ratio the test fixtures use, big enough that pair loss is a real signal.
    s.n_layer, s.n_head, s.n_embd, s.block_size = 2, 2, 64, 128
    s.batch_size = 8
    s.checkpoint_every = CHECKPOINT_EVERY
    s.post_min_pairs = 0
    s.post_rehearsal = 0.0  # no corpus at all -- nothing to rehearse on
    s.post_gate_margin = -1  # gate needs corpus val; irrelevant here, keep it inert
    s.post_learning_rate = 3e-4
    s.train_threads = 3
    return s


def _seed_pretrained(settings: Settings, seed: int) -> None:
    import torch

    from babble.model import Babbler, config_from_settings
    from babble.trainer import _build_optimizer, save_checkpoint

    torch.manual_seed(seed)
    device = force_cpu_device()
    model = Babbler(config_from_settings(settings)).to(device)
    optimizer = _build_optimizer(model, settings)
    save_checkpoint(settings, model, optimizer, 1, 4.2)


def _shuffle_words(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def build_shuffled_control() -> list[AugmentedPair]:
    """Same generated variants, word order destroyed within each half --
    same size, same vocabulary, no corpus/paraphraser word order. If
    dilution/regularisation is the whole story here too (the corpus-level
    generator's finding, PIPELINE_REVAMP_2026-08-20.md §7.2), this should
    match or beat the real-order variants."""
    real = AugmentedPairStore(REAL_DATA / "augmented_pairs.jsonl").all()
    rng = random.Random(1234)
    shuffled = []
    for row in real:
        shuffled.append(
            AugmentedPair(
                id="shuf-" + row.id,
                prompt=_shuffle_words(row.prompt, rng),
                chosen=_shuffle_words(row.chosen, rng),
                source_pair_id=row.source_pair_id,
                variant_index=row.variant_index,
            )
        )
    return shuffled


def _cell_dir(name: str, seed: int) -> Path:
    return GRID_ROOT / f"{name}-s{seed}"


def run_cell(name: str, seed: int, augmented_pairs: list[AugmentedPair], pretrained_src: Path) -> dict:
    root = _cell_dir(name, seed)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    settings = _tiny_model_settings(root)
    settings.ensure_dirs()

    # Same real pairs (and the same train/val split, since it is a pure
    # function of pair id) in every cell -- only the augmented pool varies.
    shutil.copyfile(REAL_DATA / "interactions.jsonl", settings.interactions_path)
    shutil.copyfile(REAL_DATA / "consent.json", settings.consent_path)
    if (REAL_DATA / ".salt").exists():
        shutil.copyfile(REAL_DATA / ".salt", settings.salt_path)

    shutil.copyfile(pretrained_src, settings.latest_checkpoint)
    AugmentedPairStore(settings.augmented_pairs_path).extend(augmented_pairs)

    result = post_train(
        settings, force=True, steps=STEPS, seed=seed, echo=False,
        include_pair_augmentation=bool(augmented_pairs),
    )
    cell = {
        "cell": name,
        "seed": seed,
        "augmented_pairs": len(augmented_pairs),
        "pairs_trained": result.pairs_trained,
        "final_step": result.final_step,
        "pair_val": round(result.val_loss, 4) if result.val_loss is not None else None,
    }
    print(json.dumps(cell), flush=True)
    return cell


def main() -> int:
    if not (REAL_DATA / "augmented_pairs.jsonl").exists():
        print("run `python -m experiments.pairaugment_generate` first", file=sys.stderr)
        return 1

    GRID_ROOT.mkdir(parents=True, exist_ok=True)
    real_variants = AugmentedPairStore(REAL_DATA / "augmented_pairs.jsonl").all()
    shuffled_variants = build_shuffled_control()

    def by_source(variants: list[AugmentedPair]) -> dict[str, list[AugmentedPair]]:
        out: dict[str, list[AugmentedPair]] = {}
        for v in variants:
            out.setdefault(v.source_pair_id, []).append(v)
        for source_id in out:
            out[source_id].sort(key=lambda v: v.variant_index)
        return out

    real_by_source = by_source(real_variants)
    shuf_by_source = by_source(shuffled_variants)

    def take_n(grouped: dict[str, list[AugmentedPair]], n: int) -> list[AugmentedPair]:
        return [v for group in grouped.values() for v in group[:n]]

    # A shared pretrained snapshot per seed, reused by every cell at that
    # seed so the only thing that differs between cells is the augmented
    # pool -- same discipline `experiments/post_grid.py` uses.
    pretrained_by_seed: dict[int, Path] = {}
    for seed in SEEDS:
        p_dir = GRID_ROOT / f"pretrained-s{seed}"
        if p_dir.exists():
            shutil.rmtree(p_dir)
        p_dir.mkdir(parents=True)
        settings = _tiny_model_settings(p_dir)
        settings.ensure_dirs()
        _seed_pretrained(settings, seed)
        pretrained_by_seed[seed] = settings.latest_checkpoint

    cells = [
        ("baseline-0x", lambda: []),
        ("aug-1x", lambda: take_n(real_by_source, 1)),
        ("aug-3x", lambda: take_n(real_by_source, 3)),
        ("aug-5x", lambda: take_n(real_by_source, 5)),
        ("aug-3x-shuffled", lambda: take_n(shuf_by_source, 3)),
    ]

    results = []
    for name, builder in cells:
        for seed in SEEDS:
            results.append(run_cell(name, seed, builder(), pretrained_by_seed[seed]))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "grid.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for cell in results:
            fh.write(json.dumps(cell) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
