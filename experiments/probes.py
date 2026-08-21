"""Fixed-probe comparison across checkpoints: same prompts, same seed, same
sampler -- so two checkpoints' outputs differ only by their weights.

Also scores each checkpoint on the trainer's own held-out corpus val split,
so the sample sits next to the number that is supposed to summarise it.

Usage:
    python -m experiments.probes ckpt1.pt ckpt2.pt ... [--label a --label b ...]
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from babble.config import Settings
from babble.cpu_runtime import force_cpu_device
from babble.generate import continue_text
from babble.model import Babbler, ModelConfig
from babble.trainer import corpus_rows, eval_loss, split_rows, to_examples

PROBES = ("hello", "the cat", "why is", "boop")


def load(path: str) -> tuple[Babbler, int]:
    device = force_cpu_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload.get("step", -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--label", action="append", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--samples", type=int, default=2, help="samples per probe")
    args = ap.parse_args()

    labels = args.label or [p.rsplit("/", 1)[-1] for p in args.checkpoints]
    settings = Settings.from_env()
    rows = corpus_rows(settings)
    split = split_rows(rows, settings)

    for path, label in zip(args.checkpoints, labels):
        model, step = load(path)
        val_examples = to_examples(split.val, model.config.block_size)
        train_examples = to_examples(split.train, model.config.block_size)
        report = {
            "label": label,
            "step": step,
            "params": model.num_params(),
            "corpus_val": round(eval_loss(model, val_examples), 4) if val_examples else None,
            "corpus_train": round(eval_loss(model, train_examples), 4) if train_examples else None,
            "probes": {},
        }
        for probe in PROBES:
            outs = []
            for i in range(args.samples):
                torch.manual_seed(args.seed + i)
                outs.append(
                    continue_text(
                        model, probe, max_new_tokens=64,
                        temperature=settings.temperature, top_k=settings.top_k,
                    )
                )
            report["probes"][probe] = outs
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
