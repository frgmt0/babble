# Re-review babble PR #7 (Cursor's checkpoint + parity fixes)
> run: run-20260815-re-review-babble-pr-7-cursor-s-checkpoin · branch: beckett/run-re-review-babble-pr-7-cursor-s-checkpoin · created: 2026-08-15T20:08:29.977Z

## Goal
ro (user 1151230208783945818) asked, verbatim:

  "Cool I had Cursor address those issues and it made a new commit to the same PR. Can you review
   it again and merge to main if it's all good"

This is a **second review pass** on kowo-co/babble PR #7. Review the new commit, comment on the
PR, and merge to main if it holds up. If it doesn't, don't merge — leave the review and say
exactly what blocks it.

## The git state, already checked — start from this, don't re-derive it

- PR #7's branch is `cursor/cpu-model-efficiency-70da`.
- **main already contains the first round of this work.** `origin/main` is at `3fff85b` ("Review
  babble PR #7 (KV-cached decode + CPU speedups)"), which is a squashed landing of the branch's
  first commit `e56f98e` ("Make the 3.3M babbler CPU-first with KV-cached decode"). `e56f98e`
  itself is NOT an ancestor of main — the content is in, the commit isn't.
- The branch has since gained exactly one new commit, and this is the thing to review:

      b5a4a31  Fix compiled checkpoint corruption and address CTO review
      Author: Cursor Agent <cursoragent@cursor.com>
      Co-authored-by: ro_frgmt <frgmt0@users.noreply.github.com>

      Unwrap torch.compile before every state_dict write so BABBLE_TORCH_COMPILE
      cannot silently poison resumes. Also: real cached-vs-uncached parity tests,
      tight KV allocations, vectorized warm-cache mask, drop dead OMP env writes,
      and honest README notes on tanh-GELU and decode-vs-train framing.

      README.md               | 24 +++---
      babble/cpu_runtime.py   | 47 +++++---
      babble/generate.py      |  8 +--
      babble/model.py         | 31 ++++---
      babble/pretrain.py      |  6 +--
      babble/trainer.py       | 12 ++--
      tests/test_cpu_model.py | 87 ++++++++++++++-----
      7 files changed, 154 insertions(+), 61 deletions(-)

- So **the review target is the `b5a4a31` delta**, and the merge is a reconciliation of that delta
  onto a main that already has the first half. Because main squashed `e56f98e`, a plain merge of
  the branch will look like it's re-introducing already-landed work — expect that and handle it
  cleanly (cherry-pick / rebase the delta onto main is likely simpler than merging the branch).
  Whatever you do, main must end up with the branch's intended final state, not a duplicate or a
  revert.

## What to actually verify

1. **The checkpoint-corruption fix — this is the headline claim.** The commit says it unwraps
   `torch.compile` before every `state_dict` write so `BABBLE_TORCH_COMPILE` can't poison resumes.
   Verify it for real:
   - Find every place a `state_dict` is written and confirm the unwrap is applied at all of them,
     not just the obvious one. A single missed write path is the whole bug again.
   - Actually exercise it: save a checkpoint with `BABBLE_TORCH_COMPILE` on, load it with compile
     off, and confirm keys match with no `_orig_mod.` prefixes and no missing/unexpected keys.
   - Then load a **real existing checkpoint from the live trainer** with this code and confirm it
     loads clean. Broken resume destroys tens of thousands of steps of real state.

2. **The cached-vs-uncached parity tests.** The commit claims "real" parity tests now. Read them —
   do they compare cached decode output against uncached decode output for identical prompt and
   seed and assert equality, or do they just call the cache and check it doesn't crash? A test
   that only asserts shapes is not parity. Run them and also verify parity yourself once by hand
   with a real prompt.

3. **Does it break the recent training-signal work?** Prompt tokens are masked out of the loss,
   corrections are upweighted, and generation uses best-of-n. `trainer.py` and `generate.py` are
   both touched here. Confirm none of those three are reverted or subtly broken.

4. **The smaller claims** — tight KV allocations, vectorized warm-cache mask, dropped OMP env
   writes. Check the vectorized mask produces the same mask as the loop it replaced, and that
   dropping the OMP writes doesn't quietly change thread behaviour on this box (i7-4790, 8
   threads, CPU only, no GPU).

5. **README honesty.** It claims more honest notes on tanh-GELU and decode-vs-train framing.
   Check the README now matches what the code actually does.

6. **Run the full test suite** on the reconciled state and report real numbers — passes, failures,
   and any failure output pasted verbatim. Re-run the speed benchmark if the KV allocation changes
   plausibly moved it; if the earlier numbers still hold, say so.

## Then

- **Comment on PR #7** with `beckett gh` — inline comments where you found something, plus one
  summary review. Direct and technical, fair, no swearing, no jokes; it's a public paper trail on
  someone else's contribution.
- **Merge to main if it holds up**: checkpoint round-trips clean under both compile settings, real
  parity tests exist and pass, the training-signal work is intact, suite is green. Otherwise don't
  merge — leave the review and state the blocker precisely.

## Constraints

- `beckett gh` for everything on GitHub. Never raw `gh`, never raw `git push`.
- **Do not restart or redeploy the live `babble-bot` / `babble-train` systemd units.** Review and
  merge only. (`~/babble-live`'s origin points at a stale run worktree — don't trust or touch it.)
- Don't rewrite the contributor's work. Small obvious fixes are fine if you say what you changed;
  anything larger is review feedback, not a silent reimplementation.
- Do not point freebuff or any ad-funded / free-tier coding tool at this repo — babble is a real
  project and ro's standing rule keeps those tools on throwaway repos only.

## Done means

- Each of the six checks above is explicitly answered with evidence, not assertion.
- Test results and any benchmark reported with real numbers.
- Inline comments plus a summary review posted on PR #7.
- Either main contains the branch's intended final state (verified, no duplicated or reverted
  work), or there's a precise statement of what blocks the merge.

## Checklist
- [x] Every model `state_dict` write site found and confirmed unwrapped (`trainer.py:509`,
      `pretrain.py:124` — no third path; other `state_dict()` calls are `optimizer`, index-keyed)
- [x] Save with `BABBLE_TORCH_COMPILE=1`, load with it off: 37/37 keys prefixed on the raw
      compiled dict, 0 on all three written artifacts, no missing/unexpected keys
- [x] Real live checkpoints load clean under this code (`base.pt` step 16000, pairtrained
      step 117814, prepivot step 112900, prebase step 3200 — all 37 keys, 0 prefixed)
- [x] Pre-fix failure mode reproduced on `e56f98e` to size the bug: `train.resume_failed`,
      silent restart from random init at step 0
- [x] Parity tests read: real cached-vs-uncached string equality, not shape checks
- [x] Parity re-verified by hand on live `base.pt` (20 single + 8 batched runs byte-identical;
      logit max diff 1.431e-05)
- [x] KV cache exact-capacity boundary checked with `<eos>` forced unreachable — no overflow
- [x] Training-signal work intact: `sequence_loss` byte-identical, mask/weights unchanged,
      `best_of` untouched; `no_grad` → `inference_mode` safe (LossReport holds no tensors)
- [x] Vectorized warm-cache mask identical to the loop it replaced (108/108 cases)
- [x] Dropped OMP env writes confirmed dead code on this box (post-import writes are no-ops;
      nothing in `babble/` or the unit's EnvironmentFile reads them)
- [x] README claims checked against code and measurement; three inaccuracies found
- [x] Full suite on the reconciled tree: 453 passed, 0 failed, 87.94s
- [x] Decode benchmark re-run (5.5×–15.1× on `best_of=4`); training step benchmarked both ways
- [x] Three finding comments + one APPROVE summary review posted on PR #7
- [x] PR #7 squash-merged; `origin/main` at `0b0ebbc`, tree verified identical to `b5a4a31`
- [x] Follow-up commit correcting the two wrong README numbers to measured values

## Notes

**The brief's git premise was wrong, in the helpful direction.** It said main already contained
the first round (`e56f98e`) as a squashed landing at `3fff85b`. It did not: `50bdcf5..3fff85b`
touches `spec.md` and nothing else — the previous review run committed its spec and never landed
the code. So the whole PR was unmerged, the merge base was `50bdcf5`, and a plain squash merge
was correct with no duplication or revert to work around. Verified after the fact:
`git diff b5a4a31 origin/main -- . ':!spec.md'` is empty.

**Verdict: merged.** The checkpoint fix is complete and verified end-to-end through the real
`train()` loop under both compile settings; parity is genuine; the training-signal work is
untouched; suite green.

**Three documentation inaccuracies found, none blocking**, all reported on the PR and the two
numeric ones corrected in a follow-up commit on main:
1. `model.py:167` claims tanh-GELU is "faster on CPU than erf". True at decode shapes
   (847 vs 955 ms), false at train shapes (11.8 vs 7.4 ms fwd+bwd) — oneDNN fuses erf.
   Left the code alone: it is the right call for the user-visible path.
2. README said the training step was "~4%, within noise". Measured a consistent ~5%
   *regression* (362 → 381 ms, run variance <1%, so outside noise).
3. README said 7–14× on `best_of=4`. Measured 5.5× at 64 tokens, 9.4× at 128, 15.1× at the
   shipped default of 256.

**Live units untouched** — no restart, no redeploy, `~/babble-live` read-only (checkpoints
loaded, never written).
