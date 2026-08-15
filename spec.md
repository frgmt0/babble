# Two-stage pretrain: base corpus + voice pass
> run: run-20260815-two-stage-pretrain-base-corpus-voice-pas · branch: beckett/run-two-stage-pretrain-base-corpus-voice-pas · created: 2026-08-15T06:40:22.039Z

## Goal
ro (user 1151230208783945818) asked for the two-stage pretraining pipeline for babble, and said "okay run it".

## Background, in ro's words

The bot collects an unlabelled pretraining corpus from Discord (consented rows only). It currently
has ~77 rows / ~2,000 characters against a 3.28M-param char model, so it memorizes the corpus and
composes nothing new. ro: "we are trying to make a fun model people talk to get wacky sentences
that are reflected in themselves" — so the human rows must stay, and must be the LAST thing the
model trains on so the voice sticks.

Earlier in the conversation ro also said: "id prefer to keep the human data", "maybe we should
re-pretrain every 100 contributions maybe?", "just make sure the new checkpoint is loaded into the
new bot instance", "increase the outpout length/context window concsiderably", "id say like 512 for
blocksize and like 256 output?", and "yeah if we need to run another pretrain thats fine".

The agreed design (already discussed and accepted on channel):

- Stage 1 — BASE. Pretrain from random init on a large external text corpus. Two sources:
  (a) an English word list, for word shape/spelling; (b) TinyStories-style simple short stories,
  for sentence structure. Stories are the part that actually teaches grammar; the word list alone
  cannot, because a word list has no sentences in it.
- Freeze the resulting base checkpoint as a distinct, reusable artifact on disk. It never gets
  overwritten by the human pass.
- Stage 2 — VOICE. Continue-training FROM the frozen base checkpoint on the consented human corpus
  rows only. This is cheap (seconds) and reruns from the clean base every time, so nothing
  compounds across reruns.
- No continuous training loop. The old `babble train --loop` burned CPU for nothing — it ran to
  step 3,200 with loss flat at ~0.238 since step 2,000. I already stopped it. Stage 2 fires on a
  trigger instead: every 100 new corpus rows, or on demand from the CLI.

## Repo

`kowo-co/babble`. The live install is a separate checkout at `~/babble-live` (systemd user units
`babble-bot` and `babble-train`) — do NOT edit or deploy to `~/babble-live`, just build and land
the change in the repo. Relevant existing files: `babble/config.py` (has `block_size: int = 256`
and `max_new_tokens: int = 96`, both env-overridable via `BABBLE_BLOCK_SIZE` /
`BABBLE_MAX_NEW_TOKENS`), `babble/cli.py`, `babble/core.py`, `babble/tokenizer.py`
(`block_size`-derived budgets), `babble/train*`, `checkpoints/` (`latest.pt`, `ckpt-*.pt`,
`loss.jsonl`).

## What to build

1. **External corpus loaders.** A way to load base-stage text that is NOT user data and therefore
   does NOT go through the consent path at all — dictionary words and stories are nobody's
   messages. Keep it a clearly separate corpus source from the consented human rows. The human
   rows must stay individually deletable exactly as they are today; do not fold them into the
   external source, do not change the consent model, and do not change what the bot collects.
   - Word list: prefer a real English word list. `/usr/share/dict/cracklib-small` exists on this
     box; a larger list fetched at prepare-time is better if you can get one reliably. Cache it to
     disk so the download happens once.
   - Stories: TinyStories (roneneldan/TinyStories on Hugging Face) or an equivalent simple-English
     story corpus. Cache to disk. Make the target size configurable and pick a sane default —
     the rough rule is ~20 tokens per parameter, so 3.28M params wants on the order of tens of
     millions of characters; if that is impractical to download or train on this box, take as much
     as is practical, say so in the README, and make the amount a flag rather than silently
     truncating.
   - If a download is unavailable in the run environment, the code must still be correct and
     testable with a small fixture, and the prepare step must fail loudly rather than silently
     training on nothing.

2. **Stage 1 CLI: base pretrain.** A command that wipes/ignores existing weights, trains from
   random init on the external corpus at the new geometry, and writes the result as a frozen base
   checkpoint under a stable path (e.g. `checkpoints/base.pt`) that stage 2 reads and never
   overwrites. Report loss/val as the existing trainer does.

3. **Stage 2 CLI: voice pass.** A command that loads the frozen base checkpoint and continues
   training on the consented human corpus rows only, then writes `latest.pt` (what the bot serves).
   Must be safe to rerun repeatedly — always from base, never from the previous voice checkpoint.

4. **Trigger, not a loop.** Stage 2 fires when the corpus has grown by 100 rows since the last
   voice pass, or on demand. Persist the last-trained row count so a restart doesn't re-fire.
   Make the threshold configurable. Do not reintroduce a continuous cycling loop.

5. **Geometry change.** `block_size` 256 → 512 and `max_new_tokens` 96 → 256, per ro. Note that
   changing `block_size` invalidates every existing checkpoint (positional embeddings), which is
   why it lands with the base retrain and not before. Check `babble/tokenizer.py` budget helpers
   and anything else derived from `block_size` still behaves at 512. Archive, don't delete, the
   existing checkpoints.

6. **Bot picks up new checkpoints.** Today the bot reads `latest.pt` at boot and needs a bounce to
   see a new one. Make it pick up a newly written checkpoint without a manual restart (watch mtime
   / reload on change), and make sure a half-written checkpoint can never be loaded — the existing
   `.partial` staging directory suggests the write path already stages; use or extend that.

7. **Collection feed unaffected.** The Discord channel currently reports corpus rows as they land.
   Don't break it. If a training stage produces something worth posting, keep it consistent with
   what's there.

## Constraints

- Consent model is untouched and stays failing-closed. External corpus never enters the consented
  human corpus, and human rows never get merged into the external source.
- Don't touch `~/babble-live` or the running bot. Land the change in the repo.
- Keep it in the existing style (Python, uv, the existing config/env-override pattern).
- Update the README/spec so the two-stage flow and the new commands are documented.

## Done means

- `bun`-equivalent: the project's own test suite passes (`uv run pytest` or whatever the repo
  uses), including new tests for the external loaders, the two-stage checkpoint handling (base is
  never overwritten by the voice pass), the 100-row trigger, and the 512 geometry.
- A base pretrain command exists and runs end to end on a small fixture corpus in tests.
- A voice pass command exists that starts from the frozen base every time and writes `latest.pt`.
- `block_size` is 512 and `max_new_tokens` is 256 by default, with old checkpoints archived.
- The bot loads a newly written `latest.pt` without a manual restart.
- README documents: stage 1, stage 2, the trigger, and how to rerun.

## Ceiling

Build the pipeline and land it. Don't redesign the model architecture beyond the block_size/output
changes, don't touch the consent or collection code paths beyond what's needed, and don't run a
multi-hour real pretrain as part of this run — the actual base training run is a separate step
once the pipeline exists.

## Checklist
- [x] External corpus loaders module (word list + stories), cached to disk, fixture-testable, fails loudly on empty — `babble/external.py`
- [x] `babble prepare-base` fetches/caches external corpus, configurable size (`--story-chars`/`--word-limit`), no consent path
- [x] Stage 1 CLI `babble base-pretrain`: random init on external corpus at 512 geometry -> `checkpoints/base.pt` (frozen, never overwritten)
- [x] Stage 2 CLI `babble voice-pass`: continues from `base.pt` on consented human corpus only -> `latest.pt`; safe to rerun (always from base)
- [x] Trigger: voice pass fires at +N rows (default 100) or `--force`; last-trained count persisted in `voice_state.json`; no loop
- [x] Geometry: block_size 256->512, max_new_tokens 96->256 defaults; tokenizer budgets verified at 512
- [x] Archive (not delete) existing checkpoints on base retrain — `checkpoints/archive/<ts>/`
- [x] Bot hot-reloads new latest.pt without restart (existing CheckpointGenerator mtime watch); half-written never loadable (.partial staging)
- [x] Collection feed untouched (no edits to discord_feed collection path); trigger wired via injected `_VoiceTrigger`, mirrors publisher
- [x] Consent model untouched, fails closed; external corpus never enters human corpus and vice versa (separate paths, tested)
- [x] Tests: loaders, base-never-overwritten, 100-row trigger, 512 geometry — `uv run --extra dev pytest` = 442 passed
- [x] README documents stage 1, stage 2, trigger, rerun (new "Two-stage pretraining" section)

## Notes
(worker scratch: decisions, blockers, handoff notes)
- torch 2.13.0+cpu present; deps: torch, discord.py, huggingface_hub (NO datasets/numpy).
- /usr/share/dict/cracklib-small: after junk filter -> ~54,403 usable words. TinyStories real download verified (V2 valid split, ~22M chars).
- Bot already hot-reloads latest.pt via CheckpointGenerator._ensure_current (mtime) — spec's "needs a bounce" was already solved; verified and left as-is.
- Test runner: `uv run --extra dev python -m pytest` (bare `pytest` isn't on PATH).
- New modules: babble/external.py (loaders), babble/pretrain.py (stages + trigger + VoiceAutoTrigger). CLI: prepare-base, base-pretrain, voice-pass, voice-status.
- Design: kept stage functions OUT of trainer.py's complex loop; reused its low-level helpers (make_batch/save_checkpoint/eval_loss/append_curve/be_polite). Voice uses fresh AdamW + save_checkpoint (writes latest.pt + ckpt-*.pt); base uses _save_to base.pt only. prune globs ckpt-*.pt so base.pt is safe.
- Did NOT run a real multi-hour base pretrain (ceiling). Pipeline verified on fixtures + tiny 512-geometry model. Real base run is a separate step.
