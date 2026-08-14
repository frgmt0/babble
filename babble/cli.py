"""`babble <command>` -- everything you can do without a Discord token.

    babble fake-data          write a dozen made-up corrections to play with
    babble train --loop       the polite background trainer
    babble sample -p hello    generate from the latest checkpoint
    babble curve              the loss curve, as a picture
    babble summary            one-shot state of the whole thing
    babble logs --follow      watch it live (read-only, never mutates)
    babble export             build the HuggingFace dataset directory
    babble bot                connect to Discord (needs BABBLE_DISCORD_TOKEN)
<<<<<<< ours
=======
    babble rescan-blocklist   purge stored rows that now match the blocklist
>>>>>>> theirs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="babble",
        description="A tiny from-scratch model that learns to talk from Discord corrections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("bot", help="run the Discord bot (needs BABBLE_DISCORD_TOKEN)")

    train = sub.add_parser("train", help="train from the stored corrections")
    train.add_argument("--steps", type=int, default=None, help="steps per cycle")
    train.add_argument("--loop", action="store_true", help="keep cycling: work, rest, repeat")
    train.add_argument("--cycles", type=int, default=None, help="stop after N cycles")
    train.add_argument("--seed", type=int, default=None, help="deterministic run")
    train.add_argument("--quiet", action="store_true", help="no per-checkpoint printing")

    gen = sub.add_parser("sample", help="generate from the latest checkpoint")
    gen.add_argument("-p", "--prompt", default="hello")
    gen.add_argument("-n", "--tokens", type=int, default=None)
    gen.add_argument("--temperature", type=float, default=None)
    gen.add_argument("--top-k", type=int, default=None)
    gen.add_argument("-c", "--count", type=int, default=1, help="how many samples")

    sub.add_parser("curve", help="print the loss curve")
    sub.add_parser("summary", help="step, loss, checkpoints, consent and row counts")

    logs = sub.add_parser("logs", help="read the event log (never modifies it)")
    logs.add_argument("-n", "--lines", type=int, default=40)
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("--json", action="store_true", help="the structured log instead of the prose one")

    export = sub.add_parser("export", help="build the HuggingFace dataset directory")
    export.add_argument("--out", type=Path, default=None)
    export.add_argument("--push", action="store_true", help="upload it (needs HF_TOKEN)")
    export.add_argument("--repo", default=None, help="dataset repo id")
    export.add_argument("--private", action="store_true")

    fake = sub.add_parser("fake-data", help="seed made-up corrections for offline testing")
    fake.add_argument("--user", default=None, help="fake user id to attribute them to")

<<<<<<< ours
=======
    sub.add_parser(
        "rescan-blocklist",
        help="purge stored rows that now match the content blocklist",
    )

>>>>>>> theirs
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    if args.command is None:
        build_parser().print_help()
        return 0

    if args.command == "bot":
        from .bot import run_bot

        return run_bot(settings)

    if args.command == "train":
        from .trainer import train

        result = train(
            settings,
            steps=args.steps,
            loop=args.loop,
            max_cycles=args.cycles,
            echo=not args.quiet,
            seed=args.seed,
        )
        if result.steps_run == 0 and result.stopped_because == "no_data":
            print(
                "Nothing to train on yet — no consented rows.\n"
                "Try `babble fake-data` to make some up, or run the bot and get corrected.",
                flush=True,
            )
        else:
            print(
                f"ran {result.steps_run} steps to step {result.final_step}, "
                f"{result.checkpoints_written} checkpoint(s), stopped: {result.stopped_because}",
                flush=True,
            )
        return 0

    if args.command == "sample":
        from .generate import load_model, sample

        model, step = load_model(settings)
        source = "checkpoint" if settings.latest_checkpoint.exists() else "random init (no checkpoint yet)"
        print(f"# step {step} · {source} · {model.num_params():,} params", flush=True)
        for _ in range(max(1, args.count)):
            text = sample(
                model,
                args.prompt,
                max_new_tokens=args.tokens or settings.max_new_tokens,
                temperature=settings.temperature if args.temperature is None else args.temperature,
                top_k=settings.top_k if args.top_k is None else args.top_k,
            )
            print(f"{args.prompt!r} -> {text!r}", flush=True)
        return 0

    if args.command == "curve":
        from .stats import loss_history, render_curve

        print(render_curve(loss_history(settings)), flush=True)
        return 0

    if args.command == "summary":
        from .stats import loss_history, render_snapshot, snapshot

        snap = snapshot(settings)
        print(render_snapshot(snap, markdown=False), flush=True)
        history = loss_history(settings)
        if history:
            recent = ", ".join(f"{h['step']}:{h['loss']:.3f}" for h in history[-5:])
            print(f"recent checkpoints  {recent}", flush=True)
            print(f"latest sample       {history[-1].get('sample', '')!r}", flush=True)
        print(f"logs                {snap.log_bytes:,} bytes in {settings.log_dir}", flush=True)
        return 0

    if args.command == "logs":
        from .logs import follow, tail

        path = settings.log_dir / ("babble.jsonl" if args.json else "babble.log")
        for line in tail(path, args.lines):
            print(line, flush=True)
        if args.follow:
            try:
                for line in follow(path):
                    print(line, flush=True)
            except KeyboardInterrupt:
                pass
        return 0

    if args.command == "export":
        from .export_hf import ExportBlocked, build_export, push
        from .identity import Pseudonymiser
        from .logs import EventLog

        log = EventLog(settings, Pseudonymiser.load(settings), component="export")
        try:
            result = build_export(settings, out_dir=args.out, log=log)
        except ExportBlocked as exc:
            log.event("export.blocked", error=str(exc))
            print(f"export blocked: {exc}", file=sys.stderr, flush=True)
            return 1

        print(
            f"wrote {result.rows} rows ({result.corrections} corrections, "
            f"{result.approvals} 👍) to {result.path}",
            flush=True,
        )
        if result.excluded_no_consent:
            print(f"  excluded {result.excluded_no_consent} row(s): no consent", flush=True)
        if result.dropped_leaky:
            print(f"  dropped {result.dropped_leaky} row(s): contained a raw id", flush=True)
<<<<<<< ours
=======
        if result.dropped_blocklist:
            print(f"  dropped {result.dropped_blocklist} row(s): matched the content filter", flush=True)
>>>>>>> theirs

        if args.push:
            repo = args.repo or settings.hf_repo
            try:
                url = push(settings, repo, result.path, log=log, private=args.private)
            except Exception as exc:
                log.event("export.push_failed", error=f"{type(exc).__name__}: {exc}")
                print(f"push failed: {exc}", file=sys.stderr, flush=True)
                return 1
            print(f"pushed to {url}", flush=True)
        else:
            print("  (not pushed — add --push to upload)", flush=True)
        log.close()
        return 0

    if args.command == "fake-data":
        from .fakedata import FAKE_USER, seed_fake_data
        from .identity import Pseudonymiser
        from .logs import EventLog

        log = EventLog(settings, Pseudonymiser.load(settings), component="cli")
        added = seed_fake_data(settings, log=log, user_id=args.user or FAKE_USER)
        print(f"added {added} fake row(s) to {settings.interactions_path}", flush=True)
        print("now try: babble train --steps 100", flush=True)
        log.close()
        return 0

<<<<<<< ours
=======
    if args.command == "rescan-blocklist":
        from .blocklist import Blocklist
        from .identity import Pseudonymiser
        from .logs import EventLog
        from .store import InteractionStore

        log = EventLog(settings, Pseudonymiser.load(settings), component="cli")
        blocklist = Blocklist.load()
        store = InteractionStore(settings.interactions_path)
        removed = store.purge(lambda r: blocklist.matches(r.prompt, r.chosen, r.rejected))
        log.event("blocklist.rescan", terms=len(blocklist.terms), purged=removed)
        print(f"rescanned against {len(blocklist.terms)} term(s), purged {removed} row(s)", flush=True)
        log.close()
        return 0

>>>>>>> theirs
    build_parser().print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
