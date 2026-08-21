"""`babble <command>` -- everything you can do without a Discord token.

    babble fake-data          write made-up rows into both stores to play with
    babble backfill-corpus    flatten the stored correction pairs into the corpus
    babble train --force      STAGE 1: pretrain from random init on the human corpus
    babble train-status       rows since the last run, whether the trigger is due
    babble post-train --force STAGE 2: fine-tune the pretrained checkpoint on correction pairs
    babble post-status        pairs since the last post-train, whether the trigger is due
    babble synth-generate     postulate prompts for reply-shaped corpus rows -> synthetic_pairs.jsonl
    babble synth-status       synthetic vs human correction-pair counts
    babble augment-pairs      LLM-paraphrase train-side correction pairs -> augmented_pairs.jsonl
    babble augment-check      re-run the train/val leakage check on augmented_pairs.jsonl
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

    train = sub.add_parser(
        "train",
        help="pretrain from random init on the consented human corpus -> checkpoints/latest.pt",
    )
    train.add_argument("--force", action="store_true", help="run even if the +N-row trigger is not due")
    train.add_argument(
        "--steps", type=int, default=None,
        help="step CEILING, not a target -- the best-val checkpoint may win earlier (default: BABBLE_TRAIN_STEPS)",
    )
    train.add_argument(
        "--patience", type=int, default=None,
        help="stop after N non-improving checkpoint intervals, 0 = never (default: BABBLE_TRAIN_PATIENCE)",
    )
    train.add_argument("--seed", type=int, default=1, help="deterministic run")
    train.add_argument("--quiet", action="store_true", help="no per-checkpoint printing")

    sub.add_parser("train-status", help="show the training trigger state (rows since the last run)")

    post = sub.add_parser(
        "post-train",
        help="STAGE 2: fine-tune the pretrained checkpoint on the correction pairs -> latest.pt",
    )
    post.add_argument("--force", action="store_true", help="run even if the +N-pair trigger is not due")
    post.add_argument(
        "--steps", type=int, default=None,
        help="step CEILING, not a target -- the best-val checkpoint may win earlier (default: BABBLE_POST_STEPS)",
    )
    post.add_argument(
        "--patience", type=int, default=None,
        help="stop after N non-improving checkpoint intervals, 0 = never (default: BABBLE_POST_PATIENCE)",
    )
    post.add_argument("--seed", type=int, default=1, help="deterministic run")
    post.add_argument("--quiet", action="store_true", help="no per-checkpoint printing")
    post.add_argument(
        "--include-synthetic", action="store_true",
        help="also fine-tune on data/synthetic_pairs.jsonl (see `babble synth-generate`), "
        "stored and counted separately from human corrections until this is passed",
    )
    post.add_argument(
        "--augment-pairs", action="store_true",
        help="also fine-tune on data/augmented_pairs.jsonl (see `babble augment-pairs`), "
        "LLM-paraphrased variants of TRAIN-SIDE correction pairs only "
        "(default: BABBLE_POST_AUGMENT_PAIRS)",
    )

    sub.add_parser("post-status", help="show the post-train trigger state (pairs since last post-train)")

    synth = sub.add_parser(
        "synth-generate",
        help="postulated-prompt + continuation-cut pairs from the corpus -> data/synthetic_pairs.jsonl",
    )
    synth.add_argument(
        "--no-postulate", action="store_true", help="skip the postulated-prompt pairs"
    )
    synth.add_argument(
        "--no-continuations", action="store_true", help="skip the continuation-cut pairs"
    )
    synth.add_argument(
        "--cuts", type=int, default=2, help="continuation cut points per corpus row (default 2)"
    )

    synthc = sub.add_parser(
        "synth-corpus",
        help="recombine corpus phrasing into labelled synthetic rows -> data/synthetic_corpus.jsonl",
    )
    synthc.add_argument("--count", type=int, default=400, help="how many rows to sample (default 400)")
    synthc.add_argument("--seed", type=int, default=0, help="deterministic generation")
    synthc.add_argument(
        "--rebuild", action="store_true",
        help="replace the file from the current consented corpus instead of appending",
    )
    synthc.add_argument(
        "--include-val-sources", action="store_true",
        help="also build the chain from val-side rows (leaks held-out phrasing "
        "into training-side synthetic text; experiments only)",
    )

    sub.add_parser("synth-status", help="show synthetic vs human data counts")

    aug = sub.add_parser(
        "augment-pairs",
        help="LLM-paraphrase TRAIN-SIDE correction pairs into extra post-train variants "
        "-> data/augmented_pairs.jsonl",
    )
    aug.add_argument(
        "--n", type=int, default=None,
        help="variants requested per source pair (default: BABBLE_AUGMENT_PAIRS_N)",
    )
    aug.add_argument(
        "--pair-id", default=None,
        help="only generate for this one source correction-pair id "
        "(used by the auto-trigger on a fresh correction; omit to sweep every train-side pair)",
    )
    aug.add_argument(
        "--workers", type=int, default=1, help="concurrent paraphrase calls (default 1, sequential)"
    )
    aug.add_argument("--quiet", action="store_true", help="no per-run printing")

    sub.add_parser(
        "augment-check",
        help="re-run the train/val leakage check against stored augmented_pairs.jsonl",
    )

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
        from .identity import Pseudonymiser
        from .logs import EventLog
        from .trainer import train

        log = EventLog(settings, Pseudonymiser.load(settings), component="trainer", echo=not args.quiet)
        try:
            result = train(
                settings, force=args.force, steps=args.steps, patience=args.patience,
                seed=args.seed, echo=not args.quiet, log=log,
            )
        finally:
            log.close()
        if result.stopped_because == "no_data":
            print(
                "Nothing to train on yet — no consented corpus rows.\n"
                "Try `babble fake-data` to make some up, `babble backfill-corpus` if you have\n"
                "old correction pairs lying around, or run the bot and talk to it.",
                flush=True,
            )
        elif result.stopped_because == "not_due":
            new = result.current_rows - result.last_trained_rows
            print(
                f"Not due: {new} new row(s) since the last run "
                f"(threshold {settings.train_trigger_rows}). Use --force to run anyway.",
                flush=True,
            )
        else:
            val_s = f"{result.val_loss:.4f}" if result.val_loss is not None else "n/a"
            early = " (stopped early)" if result.stopped_early else ""
            print(
                f"trained {result.rows_trained} human row(s) from random init, "
                f"best checkpoint at step {result.final_step} (loss {result.last_loss:.4f}, val {val_s}) "
                f"after {result.steps_run}/{result.budget} steps{early} -> {settings.latest_checkpoint}",
                flush=True,
            )
        return 0

    if args.command == "train-status":
        from .trainer import train_trigger

        status = train_trigger(settings)
        print(
            f"corpus rows: {status.current_rows} · last trained at: {status.last_trained_rows} rows · "
            f"new since: {status.new_rows} · threshold: {status.threshold} · "
            f"due: {'yes' if status.due else 'no'}",
            flush=True,
        )
        return 0

    if args.command == "post-train":
        from .identity import Pseudonymiser
        from .logs import EventLog
        from .posttrain import post_train

        augment_pairs = args.augment_pairs or settings.post_augment_pairs
        log = EventLog(settings, Pseudonymiser.load(settings), component="post", echo=not args.quiet)
        try:
            result = post_train(
                settings, force=args.force, steps=args.steps, patience=args.patience,
                seed=args.seed, echo=not args.quiet, log=log,
                include_synthetic=args.include_synthetic,
                include_pair_augmentation=augment_pairs,
            )
        finally:
            log.close()
        if result.ran:
            val_s = f"{result.val_loss:.4f}" if result.val_loss is not None else "n/a"
            early = " (stopped early)" if result.stopped_early else ""
            synth_part = f" + {result.synthetic_pairs_trained} synthetic pair(s)" if args.include_synthetic else ""
            aug_part = f" + {result.augmented_pairs_trained} augmented pair(s)" if augment_pairs else ""
            print(
                f"stage 2 (post-train): fine-tuned {result.pairs_trained} correction pair(s)"
                f"{synth_part}{aug_part}, "
                f"best checkpoint at step {result.final_step} (loss {result.last_loss:.4f}, val {val_s}) "
                f"after {result.checkpoints_written} checkpoint(s){early}",
                flush=True,
            )
            if result.gated:
                print(
                    f"NOT promoted: corpus val {result.corpus_val_after:.4f} vs pretrain "
                    f"{result.corpus_val_before:.4f} (margin {settings.post_gate_margin}) — "
                    f"latest.pt left untouched",
                    flush=True,
                )
            else:
                gate_part = ""
                if result.corpus_val_after is not None and result.corpus_val_before is not None:
                    gate_part = (
                        f" (corpus val {result.corpus_val_after:.4f} vs pretrain "
                        f"{result.corpus_val_before:.4f})"
                    )
                print(f"promoted{gate_part} -> {result.path}", flush=True)
        elif result.reason == "too_few_pairs":
            print(
                f"Post-train refused: only {result.current_pairs} correction pair(s), "
                f"needs {settings.post_min_pairs} (BABBLE_POST_MIN_PAIRS). Use --force to run anyway "
                f"— the promotion gate still decides what ships.",
                flush=True,
            )
        elif result.reason == "no_pretrain":
            print("No pretrained checkpoint yet. Run `babble train --force` first.", flush=True)
        elif result.reason == "not_due":
            new = result.current_pairs - result.last_trained_pairs
            print(
                f"Post-train not due: {new} new correction pair(s) since the last run "
                f"(threshold {settings.post_trigger_pairs}). Use --force to run anyway.",
                flush=True,
            )
        elif result.reason == "no_data":
            print("Nothing to post-train on: no consented correction pairs.", flush=True)
        return 0

    if args.command == "post-status":
        from .post_state import post_trigger

        status = post_trigger(settings)
        print(
            f"correction pairs: {status.current_pairs} · last post-train at: {status.last_trained_pairs} pairs · "
            f"new since: {status.new_pairs} · threshold: {status.threshold} · "
            f"pretrained checkpoint: {'present' if settings.pretrained_checkpoint.exists() else 'not yet'} · "
            f"due: {'yes' if status.due else 'no'}",
            flush=True,
        )
        return 0

    if args.command == "synth-generate":
        from .identity import Pseudonymiser
        from .logs import EventLog
        from .synthetic import generate_synthetic_pairs

        log = EventLog(settings, Pseudonymiser.load(settings), component="synth")
        result = generate_synthetic_pairs(
            settings,
            postulate=not args.no_postulate,
            continuations=not args.no_continuations,
            cuts=args.cuts,
        )
        log.event(
            "synth.generate",
            scanned=result.scanned,
            reactive=result.reactive,
            generated=result.generated,
            generated_postulated=result.generated_postulated,
            generated_continuation=result.generated_continuation,
            skipped_duplicate=result.skipped_duplicate,
            skipped_blocklist=result.skipped_blocklist,
        )
        print(
            f"scanned {result.scanned} consented corpus row(s), "
            f"{result.reactive} read as a reply/interjection\n"
            f"generated {result.generated} new synthetic pair(s) -> {settings.synthetic_pairs_path}\n"
            f"  {result.generated_postulated} postulated-prompt, "
            f"{result.generated_continuation} continuation-cut\n"
            f"  skipped {result.skipped_duplicate}: already generated\n"
            f"  skipped {result.skipped_blocklist}: matched the content filter\n"
            f"(not trained on until `babble post-train --include-synthetic`)",
            flush=True,
        )
        log.close()
        return 0

    if args.command == "synth-corpus":
        from .identity import Pseudonymiser
        from .logs import EventLog
        from .synthcorpus import generate_synthetic_corpus

        log = EventLog(settings, Pseudonymiser.load(settings), component="synth")
        result = generate_synthetic_corpus(
            settings,
            count=args.count,
            seed=args.seed,
            rebuild=args.rebuild,
            exclude_val=not args.include_val_sources,
        )
        log.event(
            "synth.corpus",
            source_rows=result.source_rows,
            excluded_val_rows=result.excluded_val_rows,
            requested=result.requested,
            generated=result.generated,
            stored_total=result.stored_total,
            skipped_real_duplicate=result.skipped_real_duplicate,
            skipped_blocklist=result.skipped_blocklist,
            rebuild=args.rebuild,
        )
        print(
            f"recombined {result.source_rows} consented corpus row(s) "
            f"({result.excluded_val_rows} val-side row(s) excluded) -> "
            f"{result.generated} synthetic row(s) "
            f"({'rebuilt' if args.rebuild else 'appended'}, {result.stored_total} stored) "
            f"-> {settings.synthetic_corpus_path}\n"
            f"  skipped {result.skipped_real_duplicate}: replayed a real row verbatim\n"
            f"  skipped {result.skipped_blocklist}: matched the content filter\n"
            f"(mixed into the pretrain train side by default; "
            f"BABBLE_TRAIN_SYNTHETIC=0 disables)",
            flush=True,
        )
        log.close()
        return 0

    if args.command == "synth-status":
        from .post_state import pair_count
        from .synthcorpus import synthetic_row_count
        from .synthetic import synthetic_pair_count

        print(
            f"synthetic pairs: {synthetic_pair_count(settings)} · "
            f"synthetic corpus rows: {synthetic_row_count(settings)} · "
            f"human correction pairs: {pair_count(settings)} · "
            f"(pairs train only with `babble post-train --include-synthetic`; "
            f"rows mix into pretrain unless BABBLE_TRAIN_SYNTHETIC=0)",
            flush=True,
        )
        return 0

    if args.command == "augment-pairs":
        from .identity import Pseudonymiser
        from .logs import EventLog
        from .pairaugment import LeakageError, assert_no_leakage, generate_augmented_pairs

        log = EventLog(settings, Pseudonymiser.load(settings), component="augment", echo=not args.quiet)
        n = args.n or settings.augment_pairs_n
        result = generate_augmented_pairs(
            settings, n=n, max_workers=max(1, args.workers),
            pair_ids=[args.pair_id] if args.pair_id else None,
        )
        log.event(
            "augment.generate",
            source_pairs=result.source_pairs,
            train_side_pairs=result.train_side_pairs,
            val_side_pairs=result.val_side_pairs,
            requested_per_pair=result.requested_per_pair,
            generated=result.generated,
            skipped_duplicate=result.skipped_duplicate,
            skipped_blocklist=result.skipped_blocklist,
            skipped_already_covered=result.skipped_already_covered,
            failed_pairs=result.failed_pairs,
        )
        if not args.quiet:
            shown_failures = "; ".join(result.failures[:3]) + (" ..." if len(result.failures) > 3 else "")
            print(
                f"{result.train_side_pairs} train-side pair(s) eligible "
                f"({result.val_side_pairs} val-side, never touched) -> "
                f"{result.generated} new variant(s) -> {settings.augmented_pairs_path}\n"
                f"  skipped {result.skipped_already_covered}: already had {n}+ variant(s)\n"
                f"  skipped {result.skipped_duplicate}: identical text already stored\n"
                f"  skipped {result.skipped_blocklist}: matched the content filter\n"
                f"  failed {result.failed_pairs}"
                + (f": {shown_failures}" if result.failures else ""),
                flush=True,
            )
        try:
            report = assert_no_leakage(settings)
        except LeakageError as exc:
            print(f"LEAKAGE CHECK FAILED: {exc}", file=sys.stderr, flush=True)
            log.close()
            return 1
        if not args.quiet:
            print(
                f"leakage check: {report.checked} stored, {report.train_side} train-side, "
                f"0 val-side, {report.orphaned} orphaned -- clean",
                flush=True,
            )
        log.close()
        return 0

    if args.command == "augment-check":
        from .pairaugment import LeakageError, assert_no_leakage

        try:
            report = assert_no_leakage(settings)
        except LeakageError as exc:
            print(f"LEAKAGE CHECK FAILED: {exc}", file=sys.stderr, flush=True)
            return 1
        print(
            f"leakage check: {report.checked} stored, {report.train_side} train-side, "
            f"0 val-side, {report.orphaned} orphaned (source pair no longer trainable) -- clean",
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
            recent_parts = []
            for h in history[-5:]:
                part = f"{h['step']}:{h['loss']:.3f}"
                if "val_loss" in h:
                    part += f"/{h['val_loss']:.3f}"
                recent_parts.append(part)
            recent = ", ".join(recent_parts)
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
        print("now try: babble train --force --steps 100", flush=True)
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
