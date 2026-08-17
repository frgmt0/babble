"""`babble <command>` -- everything you can do without a Discord token.

    babble fake-data          write made-up rows into both stores to play with
    babble backfill-corpus    flatten the stored correction pairs into the corpus
    babble train --loop       the polite background trainer
    babble sample -p hello    continue a prefix from the latest checkpoint
    babble curve              the loss curve, as a picture
    babble summary            one-shot state of the whole thing
    babble logs --follow      watch it live (read-only, never mutates)
    babble export             build the HuggingFace dataset directory
    babble bot                connect to Discord (needs BABBLE_DISCORD_TOKEN)
    babble rescan-blocklist   purge stored rows that now match the blocklist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="babble",
        description="A tiny from-scratch model that learns to talk from a Discord corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("bot", help="run the Discord bot (needs BABBLE_DISCORD_TOKEN)")

    train = sub.add_parser("train", help="train from the stored corpus")
    train.add_argument("--steps", type=int, default=None, help="steps per cycle")
    train.add_argument("--loop", action="store_true", help="keep cycling: work, rest, repeat")
    train.add_argument("--cycles", type=int, default=None, help="stop after N cycles")
    train.add_argument("--seed", type=int, default=None, help="deterministic run")
    train.add_argument("--quiet", action="store_true", help="no per-checkpoint printing")

    prep = sub.add_parser(
        "prepare-base",
        help="download/cache the external base corpus (dictionary words + stories)",
    )
    prep.add_argument("--words", type=Path, default=None, help="local word-list file (default: BABBLE_WORDLIST_PATH)")
    prep.add_argument("--stories", type=Path, default=None, help="local stories file (skips the HF download)")
    prep.add_argument("--story-chars", type=int, default=None, help="max characters of stories to keep (0 = all)")
    prep.add_argument("--word-limit", type=int, default=None, help="max words to keep (0 = all)")

    base = sub.add_parser(
        "base-pretrain",
        help="STAGE 1: train from random init on the external corpus -> checkpoints/base.pt",
    )
    base.add_argument("--steps", type=int, default=None, help="step budget (default: BABBLE_BASE_STEPS)")
    base.add_argument("--seed", type=int, default=1, help="deterministic run")
    base.add_argument("--quiet", action="store_true", help="no per-checkpoint printing")

    voice = sub.add_parser(
        "voice-pass",
        help="STAGE 2: continue from base.pt on the human corpus -> latest.pt",
    )
    voice.add_argument("--force", action="store_true", help="run even if the +N-row trigger is not due")
    voice.add_argument(
        "--steps", type=int, default=None,
        help="step CEILING, not a target -- the best-val checkpoint may win earlier (default: BABBLE_VOICE_STEPS)",
    )
    voice.add_argument(
        "--patience", type=int, default=None,
        help="stop after N non-improving checkpoint intervals, 0 = never (default: BABBLE_VOICE_PATIENCE)",
    )
    voice.add_argument("--seed", type=int, default=1, help="deterministic run")
    voice.add_argument("--quiet", action="store_true", help="no per-checkpoint printing")

    sub.add_parser("voice-status", help="show the voice-pass trigger state (rows since last pass)")

    gen = sub.add_parser("sample", help="continue a prefix using the latest checkpoint")
    gen.add_argument("-p", "--prompt", default="hello", help="the prefix to continue from")
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

    fake = sub.add_parser("fake-data", help="seed made-up rows for offline testing")
    fake.add_argument("--user", default=None, help="fake user id to attribute them to")

    sub.add_parser(
        "backfill-corpus",
        help="flatten the stored correction pairs into the corpus (idempotent)",
    )

    sub.add_parser(
        "rescan-blocklist",
        help="purge stored rows that now match the content blocklist",
    )

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
                "Nothing to train on yet — no consented corpus rows.\n"
                "Try `babble fake-data` to make some up, `babble backfill-corpus` if you have\n"
                "old correction pairs lying around, or run the bot and talk to it.",
                flush=True,
            )
        else:
            print(
                f"ran {result.steps_run} steps to step {result.final_step}, "
                f"{result.checkpoints_written} checkpoint(s), stopped: {result.stopped_because}",
                flush=True,
            )
        return 0

    if args.command == "prepare-base":
        from .external import EmptyCorpusError, prepare_base_corpus

        try:
            result = prepare_base_corpus(
                settings,
                wordlist_path=args.words,
                stories_path=args.stories,
                word_limit=args.word_limit,
                story_chars=args.story_chars,
            )
        except EmptyCorpusError as exc:
            print(f"prepare-base failed: {exc}", flush=True)
            return 1
        print(result.summary(), flush=True)
        return 0

    if args.command == "base-pretrain":
        from .external import EmptyCorpusError
        from .identity import Pseudonymiser
        from .logs import EventLog
        from .pretrain import pretrain_base

        log = EventLog(settings, Pseudonymiser.load(settings), component="base", echo=not args.quiet)
        try:
            result = pretrain_base(settings, steps=args.steps, seed=args.seed, echo=not args.quiet, log=log)
        except EmptyCorpusError as exc:
            print(f"base-pretrain failed: {exc}\nRun `babble prepare-base` first.", flush=True)
            return 1
        finally:
            log.close()
        print(
            f"stage 1 (base): ran {result.steps_run} steps to step {result.final_step}, "
            f"{result.checkpoints_written} checkpoint(s), loss {result.last_loss:.4f} -> {result.path}",
            flush=True,
        )
        return 0

    if args.command == "voice-pass":
        from .identity import Pseudonymiser
        from .logs import EventLog
        from .pretrain import voice_pass

        log = EventLog(settings, Pseudonymiser.load(settings), component="voice", echo=not args.quiet)
        try:
            result = voice_pass(
                settings, force=args.force, steps=args.steps, patience=args.patience,
                seed=args.seed, echo=not args.quiet, log=log,
            )
        finally:
            log.close()
        if result.ran:
            stage = result.stage
            val_s = f"{stage.val_loss:.4f}" if stage.val_loss is not None else "n/a"
            early = " (stopped early)" if stage.stopped_early else ""
            print(
                f"stage 2 (voice): trained {result.rows_trained} human row(s) from base, "
                f"best checkpoint at step {stage.final_step} (loss {stage.last_loss:.4f}, val {val_s}) "
                f"after {stage.steps_run}/{stage.budget} steps{early} -> {stage.path}",
                flush=True,
            )
        elif result.reason == "no_base":
            print("No base checkpoint yet. Run `babble prepare-base` then `babble base-pretrain` first.", flush=True)
        elif result.reason == "not_due":
            new = result.current_rows - result.last_trained_rows
            print(
                f"Voice pass not due: {new} new row(s) since the last pass "
                f"(threshold {settings.voice_trigger_rows}). Use --force to run anyway.",
                flush=True,
            )
        elif result.reason == "no_data":
            print("Nothing to train on: no consented corpus rows.", flush=True)
        return 0

    if args.command == "voice-status":
        from .pretrain import voice_trigger

        status = voice_trigger(settings)
        print(
            f"corpus rows: {status.current_rows} · last voice pass at: {status.last_trained_rows} rows · "
            f"new since: {status.new_rows} · threshold: {status.threshold} · "
            f"base.pt: {'present' if status.has_base else 'MISSING'} · "
            f"due: {'yes' if status.due else 'no'}",
            flush=True,
        )
        return 0

    if args.command == "sample":
        from .generate import continue_text, load_model

        model, step = load_model(settings)
        source = "checkpoint" if settings.latest_checkpoint.exists() else "random init (no checkpoint yet)"
        print(f"# step {step} · {source} · {model.num_params():,} params", flush=True)
        for _ in range(max(1, args.count)):
            # A continuation, not an answer: the model is trained on plain text
            # and has never once seen a prompt/response boundary.
            text = continue_text(
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
        from .stats import loss_history, render_drift, render_snapshot, snapshot

        snap = snapshot(settings)
        print(render_snapshot(snap, markdown=False), flush=True)
        history = loss_history(settings)
        if history:
            recent = ", ".join(f"{h['step']}:{h['loss']:.3f}" for h in history[-5:])
            print(f"recent checkpoints  {recent}", flush=True)
            print(f"latest sample       {history[-1].get('sample', '')!r}", flush=True)
        print(f"logs                {snap.log_bytes:,} bytes in {settings.log_dir}", flush=True)
        print(f"code                {render_drift(snap)}", flush=True)
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
            f"wrote {result.rows} rows to {result.path}\n"
            f"  corpus       {result.corpus_rows} rows, {result.corpus_chars:,} chars\n"
            f"  corrections  {result.correction_rows} rows "
            f"({result.corrections} corrections, {result.approvals} 👍)",
            flush=True,
        )
        for label, consent_dropped, leaky, blocked in (
            (
                "corpus",
                result.corpus_excluded_no_consent,
                result.corpus_dropped_leaky,
                result.corpus_dropped_blocklist,
            ),
            ("corrections", result.excluded_no_consent, result.dropped_leaky, result.dropped_blocklist),
        ):
            if consent_dropped:
                print(f"  {label}: excluded {consent_dropped} row(s): no consent", flush=True)
            if leaky:
                print(f"  {label}: dropped {leaky} row(s): contained a raw id", flush=True)
            if blocked:
                print(
                    f"  {label}: dropped {blocked} row(s): matched the content filter", flush=True
                )

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
        print(f"and flattened them into {settings.corpus_path}", flush=True)
        print("now try: babble train --steps 100", flush=True)
        log.close()
        return 0

    if args.command == "backfill-corpus":
        from .backfill import backfill_corpus
        from .identity import Pseudonymiser
        from .logs import EventLog

        log = EventLog(settings, Pseudonymiser.load(settings), component="cli")
        result = backfill_corpus(settings, log=log)
        print(
            f"scanned {result.scanned} correction row(s) → "
            f"added {result.added} corpus row(s) to {settings.corpus_path}",
            flush=True,
        )
        if result.skipped_duplicate:
            print(f"  skipped {result.skipped_duplicate}: already in the corpus", flush=True)
        if result.skipped_consent:
            print(f"  skipped {result.skipped_consent}: no consent", flush=True)
        if result.skipped_blocklist:
            print(f"  skipped {result.skipped_blocklist}: matched the content filter", flush=True)
        if result.skipped_empty:
            print(f"  skipped {result.skipped_empty}: nothing to store", flush=True)
        print("  (safe to run again — it will add nothing the second time)", flush=True)
        log.close()
        return 0

    if args.command == "rescan-blocklist":
        from .blocklist import Blocklist
        from .corpus import CorpusStore
        from .identity import Pseudonymiser
        from .logs import EventLog
        from .store import InteractionStore

        log = EventLog(settings, Pseudonymiser.load(settings), component="cli")
        blocklist = Blocklist.load()
        store = InteractionStore(settings.interactions_path)
        removed = store.purge(lambda r: blocklist.matches(r.prompt, r.chosen, r.rejected))
        corpus_removed = CorpusStore(settings.corpus_path).purge(
            lambda r: blocklist.matches(r.text)
        )
        log.event(
            "blocklist.rescan",
            terms=len(blocklist.terms),
            purged=removed,
            corpus_purged=corpus_removed,
        )
        print(
            f"rescanned against {len(blocklist.terms)} term(s), purged {removed} correction "
            f"row(s) and {corpus_removed} corpus row(s)",
            flush=True,
        )
        log.close()
        return 0

    build_parser().print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
