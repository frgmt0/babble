# babble — from-scratch discord model that learns by correction
> run: run-20260814-babble-from-scratch-discord-model-that-l · branch: beckett/run-babble-from-scratch-discord-model-that-l · created: 2026-08-14T02:36:25.376Z

## Goal
ro (user 1151230208783945818, owner of Kowo) asked for this in Discord. His words, verbatim, across several messages:

"what if we make a model that learns how to talk by people interacting with it on discord. So you ping it, then you get a replied response. If you reply to that you can correct it with the response you wanted whether it be a gif and whatnot. But basically it's just got all the english words, special characters and numbers uninitialised as tokens and then it's like super small and optimal for CPU and our RAM. I think thatd be something fun."

"No but the from scratch is the most fun. Because it'll be completely random and we train on checkpoints in a consistent ultra efficient loop that doesn't make the computer unusable"

"Don't seed it from this channel. We can have it learn from interactions only when you ping get a response and then correct or react with a thumbs up 'an automated -# like this response? React with 👍. If not correct with a reply'"

"And we save this data to huggingface and so people can see. And also for security the first time we ping the bot it should make sure informed consent about the data is understood. So if you don't permit the bot to read your messages you won't be trained on."

Build this in the repo `babble` (github.com/kowo-co/babble — the repo exists and is EMPTY, so you are scaffolding it from nothing). Python, since this is a training project. Keep dependencies minimal: torch (CPU only), a Discord library, and the huggingface_hub client. No CUDA assumptions — this runs on a normal Linux box that people are also using for other things.

## What to build

**1. The model.** A tiny from-scratch transformer, a few million parameters, byte-level or
character-level tokenisation so it can emit literally anything including a URL (gif links are just
URLs — that matters, ro explicitly wants gif corrections to work). Weights start RANDOM. There is
NO pretrained base and NO seeding from any existing corpus or chat history — ro was explicit about
this twice. It starts as noise and the only data it ever sees comes from the interaction loop.
Everything must fit and train on CPU inside a couple of GB of RAM.

**2. The training loop.** Trains in the background off checkpoints on a schedule, NOT continuously
and NOT in the request path. ro's constraint, verbatim: "a consistent ultra efficient loop that
doesn't make the computer unusable" — so: niced/low-priority, a capped thread count, a configurable
step budget per cycle, and it must be safe to kill and resume from the last checkpoint at any
moment. Checkpoints are the unit of progress: each one is written to disk with its step count and
loss, and each one emits a sample generation. Expect the output to be babble for a long time —
that is the intended show, not a failure state, so the sample-per-checkpoint output and the loss
curve are first-class features, not debug output.

**3. The Discord bot.** Ping it (@mention or reply), it replies with a generation. Every bot reply
carries an automated footer in Discord subtext form, roughly:
`-# like this response? react with 👍 — if not, correct me with a reply`
Two feedback paths, both captured:
- a 👍 reaction on a bot message = weak positive signal on that response
- a reply to a bot message = a CORRECTION, the strong signal, and this is the one that matters.
Each correction is stored as a triple: what was said to the bot (the prompt), what the bot answered
(the rejected response), and what it should have said (the human's reply). A correction may be text,
a URL, a gif link, an emoji, anything — store it raw.
Make correcting the easy, obvious path and the thumbs-up the lazy fallback; a correction is worth
far more than a reaction and the UI copy should nudge that way without being preachy about it.

**4. Consent, before anything else.** The FIRST time a given user pings the bot, they get an
informed-consent prompt explaining plainly: their messages to the bot and their corrections are
stored, used to train the model, and published to a public HuggingFace dataset. They explicitly
accept or decline. Their choice is persisted per user id.
- Declined or not-yet-answered: the bot may still reply to them, but NOTHING from that user is
  stored, trained on, or exported. Fail closed — no consent record means no data, full stop.
- There must be a way for a user to withdraw consent later, and withdrawing must purge their
  already-stored rows from the local dataset (a command like `!babble consent` / `!babble forget`,
  your call on the exact surface).
- Consent state is checked at capture time, not just at reply time.

**5. HuggingFace export.** The captured triples export to a HF dataset so it is public and people
can see it (ro's ask). Only consented rows. Sensible field names, a dataset card explaining what
this is and how it was collected, and it should be safe to re-run (idempotent-ish, not duplicating
rows). Author IDs must be pseudonymised — a stable hash, never a raw Discord ID or username in the
published data. Do not push to HF automatically on every capture; make it an explicit command/script
so nothing leaks by accident.

## Constraints

- **There is no bot token yet.** ro will hand one over later. Everything must be buildable, testable
  and runnable without it: read the token from an env var, and make the model, trainer, capture
  layer and export layer testable with fakes so the whole thing can be exercised offline. Do not
  block on the token and do not fabricate one.
- Do not seed the model from Discord history, Wikipedia, or any other corpus. Random init only.
- Do not reach for a pretrained model, a tokenizer download, or an API model anywhere in the loop.
  The whole point is it learns from scratch from the interaction data.
- Keep it small and readable. This is a toy that is meant to be watched, not a framework.

## Done means

- `README.md` explains the idea, the consent model, how to run the bot and the trainer, and sets the
  honest expectation that it will babble for a long time and may never be fully coherent.
- The trainer runs on CPU from random init on a tiny sample of fake correction data, writes
  checkpoints, prints loss and a sample generation per checkpoint, and resumes correctly after being
  killed mid-run.
- The bot's reply/reaction/correction paths are covered by tests using a faked Discord layer:
  a first-time pinger gets the consent prompt; a declining user's messages are never stored; a
  consenting user's reply-correction lands as a complete triple; a 👍 lands as a positive signal.
- The HF export produces a valid dataset directory + card from stored triples, contains only
  consented rows, and contains no raw Discord IDs or usernames.
- Tests pass and the README's quickstart actually works end to end minus the token.

## Checklist

### Scaffold
- [x] 1. `pyproject.toml` with minimal deps (torch CPU, discord.py, huggingface_hub) + `[dev]` extra (pytest)
- [x] 2. `.gitignore` (data/, checkpoints/, export/, venv, __pycache__), `.env.example` with `BABBLE_DISCORD_TOKEN`
- [x] 3. Package layout `babble/` + `tests/`, `python -m babble` CLI entrypoint

### Model (random init, byte-level)
- [x] 4. `tokenizer.py`: byte-level vocab (256 bytes + `<pad> <bos> <sep> <eos>`); round-trips emoji, URLs, arbitrary unicode
- [x] 5. `model.py`: decoder-only transformer, ~3M params, CPU-only, no pretrained weights anywhere in the tree
- [x] 6. `generate.py`: temperature/top-k sampling, stops at `<eos>`, decodes bytes with `errors="replace"`
- [x] 7. Test: fresh model params are random (two seeds differ); forward+backward runs on CPU under a few hundred MB

### Store + consent (fail closed)
- [x] 8. `consent.py`: per-user-id decision store (granted/declined/unknown); unknown == no data, full stop
- [x] 9. `store.py`: append-only JSONL of triples `(prompt, rejected, chosen)`; **no raw Discord IDs on disk** — salted hashes only
- [x] 10. Capture requires consent from *every* human in the row (prompt author AND corrector/reactor)
- [x] 11. `forget(user)` withdraws consent AND purges every row where they are prompt-author or signal-author
- [x] 12. Discord mention syntax `<@123…>` scrubbed from stored content at capture time

### Discord bot
- [x] 13. `core.py`: Discord-agnostic event→action logic (pure, returns actions; no discord import)
- [x] 14. `bot.py`: thin discord.py adapter, token from `BABBLE_DISCORD_TOKEN`, never fabricated
- [x] 15. Every bot reply carries the subtext footer `-# like this response? react with 👍 — if not, correct me with a reply`
- [x] 16. Reply-to-bot = correction (strong, weight 1.0); 👍 reaction = approval (weak, weight < 1.0)
- [x] 17. Exchange log survives restart so a correction still resolves its triple after the bot reboots
- [x] 18. `!babble` commands: help / consent / accept / decline / forget / status

### Trainer
- [x] 19. `trainer.py`: resumable loop — atomic checkpoint writes, resumes step+optimizer+RNG from `latest.pt`
- [x] 20. Politeness: `os.nice(19)`, capped `torch.set_num_threads`, duty-cycled `--loop` (work then rest)
- [x] 21. Per checkpoint: append `{step, loss, sample}` to `loss.jsonl` and print the sample generation
- [x] 22. Loss masked to response tokens only; approval rows weighted lower than corrections
- [x] 23. Graceful SIGINT/SIGTERM (checkpoint then exit) and safe hard-kill (no truncated checkpoint)

### HF export
- [x] 24. `export_hf.py`: writes `export/` dataset dir + dataset card (YAML frontmatter, collection method, consent)
- [x] 25. Consent re-checked at export time; declined/unknown rows excluded even if present locally
- [x] 26. Hard guard: refuse to write if any known raw Discord ID or username appears in the output
- [x] 27. Idempotent: deterministic row ids, stable sort, re-running yields byte-identical output; push only behind explicit `--push`

### Observability (ro's follow-up: "almost log everything", non-destructively viewable)
- [x] 33. `logs.py`: append-only `logs/babble.jsonl` (structured) + `logs/babble.log` (human tail), never truncated
- [x] 34. Size-based rotation only — no truncate-on-read, no clearing on restart; reading is always side-effect free
- [x] 35. Buffered writes with a time-based flush so tailing is live but logging is never the hot path
- [x] 36. User + channel identifiers pseudonymised in logs with the *same* salted hash the export uses
- [x] 37. Content logged only for consented users; skip events log the reason + who, never the content
- [x] 38. Bot events: start / ready / ping / generation (params + checkpoint step + latency) / stop / error
- [x] 39. Consent events: prompt, accept, decline, withdraw (with rows purged)
- [x] 40. Capture events: every correction triple, every 👍, and every skipped-for-no-consent with its reason
- [x] 41. Trainer events: cycle start/stop, resume-after-kill, per-checkpoint step + loss + sample
- [x] 42. Export events: every run with counts, every push, every blocked-by-guard
- [x] 43. `babble logs [--follow]` to tail live and `babble summary` one-shot (step, last loss, checkpoints, consented users, stored triples)

### Done gates
- [x] 28. `pytest` green: consent prompt on first ping / decliner stores nothing / correction lands full triple / 👍 lands positive signal
- [x] 29. Trainer demo verified end-to-end: random init → checkpoints → killed mid-run → resumes at the right step
- [x] 30. `README.md`: the idea, consent model, run instructions, honest "it will babble for a long time" expectation
- [x] 31. README quickstart executed verbatim in a clean venv and it works, minus the token
- [x] 32. Self-review of the full diff against every "Done means" bullet; work committed

## Notes
(worker scratch: decisions, blockers, handoff notes)

**Status: complete.** 115 tests pass (`pytest`, ~6s). README quickstart verified verbatim from a
clean checkout. No Discord token was needed, used, or fabricated anywhere.

**Verified by hand, not just by tests**
- Trainer on the real 3.28M-param model: 120 steps in 8.6s on 2 threads at nice 19.
- `kill -9` mid-cycle → `latest.pt` still loaded cleanly at step 160, no torn `.tmp` files;
  next run logged `train.resume step=160` and carried on.
- `SIGTERM` → finished its step, checkpointed at 211, exited 0.
- Nothing private is in git history: no `data/`, no `consent.json`, no `.salt`, no `.env`, no tokens.

**Design decisions**
- Byte-level (not word-level) tokenizer: ro wants gif corrections, and gif links are URLs — bytes emit
  any URL, emoji or unicode with a 260-token vocab. Word tokens could not.
- Raw Discord IDs are never written to the interaction store at all (only salted hashes), so the export
  path cannot leak one even if it is buggy. Consent store is the only file keyed by real user ids.
- Consent is required from every human whose content/action is in a row. If A pings and B corrects,
  both must have accepted — A's prompt text gets published too.
- Text commands (`!babble accept`) rather than discord UI buttons: keeps the consent path inside the
  pure, testable core instead of the discord adapter.
