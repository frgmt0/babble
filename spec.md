# babble posts training cycles to a discord feed
> run: run-20260814-babble-posts-training-cycles-to-a-discor · branch: beckett/run-babble-posts-training-cycles-to-a-discor · created: 2026-08-14T05:04:40.572Z

## Goal
ro (user 1151230208783945818, owner) asked, verbatim:

"can you make it so booper (or babble for you) prints training output logs to <#1537688090217418773> like when it completes a training cyle"

Repo: `babble` (github.com/kowo-co/babble). It is already built, already on `main`, and already
running live on this box — so this is an addition to a working system, not a rewrite.

## What to build

A Discord training-log feed. When the trainer finishes a cycle / writes a checkpoint, it posts a
compact message to a configured Discord channel. Target channel id for this deployment:
`1537688090217418773`. Make it configurable via env (e.g. `BABBLE_LOG_CHANNEL_ID`) with no
hardcoded id in the code — the id above goes in `.env.example` documentation only.

What each post should carry, at minimum: cycle number, step count, current loss (and the delta from
the previous checkpoint), how many trainable rows are in the corpus, and the sample generation the
checkpoint produced. Keep it to a couple of short lines — this is a feed people watch, not a log
dump. Include the first-real-word / loss-curve character of the project: the sample is the
interesting part, so it should be prominent and rendered safely (the model emits arbitrary bytes,
so escape/truncate it and never let it break the message or ping anyone — no `@everyone`, no role
or user mentions, ever, in bot output).

Also post the notable lifecycle events to the same channel, briefly: trainer start, resume after a
kill (with the step it resumed from), and idle-because-no-consented-rows (once, not every cycle —
do not spam the channel while it sits idle).

## How it must behave

- **The trainer and the bot are separate processes.** `babble train --loop` runs standalone with no
  Discord gateway connection. Solve that properly rather than by merging the processes — either the
  trainer posts via a Discord webhook URL (simplest, no gateway needed), or it writes events to a
  file/queue the bot process tails and posts. Pick one, say why in the README, and make it work when
  the trainer is started on its own.
- **Failing to post must never disturb training.** Network error, bad channel id, missing config,
  rate limit: log it and carry on training. Posting is best-effort, always.
- **Rate-limit-aware and quiet by default.** Do not post more than one message per checkpoint, and
  provide a way to throttle (e.g. only post every Nth checkpoint) so a long run does not flood the
  channel. Sensible default that does not spam.
- **Unconfigured means silent.** No channel/webhook configured → no posting, no errors, no change in
  behavior for anyone running this without Discord.
- Respect the existing consent and pseudonymisation rules: never put raw Discord ids, usernames, or
  the content of non-consented messages into these posts.
- Everything posted must also still go to the existing `logs/babble.jsonl` + `logs/babble.log`.

## Constraints

- Keep it small and in the existing style. Do not restructure the trainer, the model, the consent
  layer or the export.
- Tests must cover it with a faked Discord/webhook layer: a checkpoint produces a well-formed post,
  a failing post does not break the training loop, unconfigured means no post, throttling works, and
  a model sample containing mention markup or `@everyone` is neutered before sending.
- Update the README (a short section on the training feed) and `.env.example`.
- `pytest` must be green.

## Done means

- With a channel/webhook configured, running `babble train --loop` posts a short, readable message
  to Discord on every checkpoint carrying cycle, step, loss + delta, row count and the sample.
- Trainer start, resume-after-kill and first-idle also post, once each, without spam.
- With nothing configured, behavior is exactly as it is today.
- A post failure is logged and training continues uninterrupted.
- Tests pass and the README documents how to point it at a channel.

## Checklist

Discord training feed:
- [x] `babble/discord_feed.py`: `TrainingFeed` (webhook-based, best-effort), `neuter_sample()`
- [x] Wired into `trainer.py`: start / resume / idle-once / checkpoint (cycle, step, loss+delta, rows, sample)
- [x] `BABBLE_LOG_WEBHOOK_URL` + `BABBLE_LOG_EVERY_N` env config, unconfigured == silent
- [x] Failing post never raises out of `_post`; training unaffected
- [x] `allowed_mentions: {parse: []}` + zero-width-space mention neutering, belt and suspenders
- [x] Tests: well-formed post, failing post, unconfigured silent, throttling, `@everyone`/mention neutered
- [x] README "Training feed" section + `.env.example`

Content blocklist (second piece of scope, folded in mid-run per ro via concierge):
- [x] `babble/blocklist.txt` (starter list) + `babble/blocklist.py` (normalise: case/diacritics/leetspeak/
      separators/repeats/spaced-letter evasion, word-boundary matching)
- [x] `BABBLE_BLOCKLIST_PATH` env override
- [x] Enforced at capture time (`core.py`: correction, approval, and the model's own output before send)
- [x] Enforced at training time (`trainer.py: consented_rows`)
- [x] Enforced at export time (`export_hf.py: select_rows` / `dropped_blocklist`)
- [x] Rejections logged as `capture.blocked` with a row fingerprint hash, never the text
- [x] User told briefly, without the term repeated back
- [x] `babble rescan-blocklist` CLI command purges existing matches (uses new `InteractionStore.purge()`)
- [x] Tests: normalisation defeats evasion, awkward-substring word NOT rejected, matching row absent from
      storage + export, rescan purge works, model-output match caught before send
- [x] README "Content filter" section, explicit "speed bump, not a guarantee" framing

Both:
- [x] `pytest` green (154/154, incl. ~33 new tests), reran 3x to check for flakiness
- [x] No hardcoded channel id / real slur literal in code paths outside `blocklist.txt` and docs

## Notes
- Repo state at spawn only had `init` commit on `main`; the actual babble implementation
  (a83d8db) lived on the sibling branch/worktree `beckett/run-babble-from-scratch-discord-model-that-l`
  and was already deployed live at `~/babble-live`. Merged that branch in (`-X ours` to keep this
  ticket's spec.md) before starting work.
- Chose a **webhook** over a file-tail relay for the training feed: trainer stays a single HTTPS
  POST with no login/gateway/heartbeat, so it degrades to "silent" rather than "half-started
  process" when unconfigured. Documented in README.
- Blocklist normalisation: separators (`.`, `-`, `_`, zero-width) are deleted outright so
  `s.l.u.r` collapses to one token; real whitespace is preserved as a word boundary (so
  `class`/`assassin` don't false-positive against `ass`) EXCEPT runs of single-letter tokens
  (`s l u r`), which are merged before matching to catch classic letter-spacing evasion.
- `InteractionStore.purge_author` refactored to share a new `purge(predicate)` with the
  blocklist rescan path — small, justified dedup, not a broader restructuring.
- Pre-existing test `test_killing_the_trainer_mid_run_leaves_a_loadable_checkpoint` is flaky
  under load (SIGKILL race against `torch.save`+`os.replace`); unrelated to this change,
  passes reliably in isolation and in repeated full-suite runs.
- Second piece of scope (content blocklist) arrived via a concierge peer message mid-run and
  was confirmed/folded into this same branch per the persona's peer-message handling rules.
