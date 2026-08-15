# Review babble PR #7 (KV-cached decode + CPU speedups)
> run: run-20260815-review-babble-pr-7-kv-cached-decode-cpu · branch: beckett/run-review-babble-pr-7-kv-cached-decode-cpu · created: 2026-08-15T19:38:44.051Z

## Goal
ro (user 1151230208783945818) asked, verbatim:

  "https://github.com/kowo-co/babble/pull/7 — can you review this PR. Comment on it, test it etc
   and merge if good"

So: a real review pass on kowo-co/babble PR #7, inline comments on the PR itself, actually run
the tests, and merge it **only if it holds up**. If it doesn't hold up, do not merge — leave the
review comments and say clearly what's blocking.

## What the PR claims

Title: "CPU-first babbler: KV-cached decode and train/infer speedups". Its own summary says it
revamps the ~3.3M-param babbler for CPU-first train and inference on the current architecture
(byte vocab, 4x256, block 512), that **checkpoint `state_dict` keys are unchanged**, and that the
KV cache is runtime-only. Author appears to be an outside contributor, not us.

## Context you need about this repo

- babble is a Discord bot plus an ML trainer, both **currently running live** on this box as
  systemd user units: `babble-bot` and `babble-train`. The live checkout is `~/babble-live`.
- The trainer is mid-run — it has been training continuously and is tens of thousands of steps
  in, resuming from checkpoints. **Anything that breaks checkpoint loading destroys real state.**
- Note that `~/babble-live`'s git origin has previously been found pointing at a stale local
  worktree path. Verify where it actually points before trusting anything about it. Do not
  deploy/restart the live units as part of this task — this run is review and merge only.
- The corpus is tiny (order 30-80 rows) and the model is ~3.3M params. Perf claims should be
  measured on this box: an i7-4790, 8 threads, 32GB, CPU only, no GPU.

## What to actually do

1. **Read the whole diff properly.** Not a skim. Pay attention to:
   - **Checkpoint compatibility.** The PR claims `state_dict` keys are unchanged. Verify that
     against an actual existing checkpoint — load a real current checkpoint with the new code and
     confirm it loads clean, no missing/unexpected keys, no silently reinitialized weights.
   - **KV cache correctness.** This is the classic place to get subtly wrong answers. Confirm that
     cached decode produces **identical** output to uncached decode for the same prompt and seed.
     If it doesn't match bit for bit, that is a blocker, and say so with the diverging output.
   - Whether it quietly changes training semantics — loss masking, sampling temperature, the
     train/val split, correction upweighting, best-of-n generation. Recent work deliberately
     masked prompt tokens out of the loss, upweighted corrections, and switched generation to
     best-of-n. **If this PR reverts or breaks any of that, it's a blocker.**
   - Anything that changes the on-disk dataset format or the consent/blocklist accounting.

2. **Run the tests.** The full suite, on the PR branch. Report the real numbers — how many pass,
   how many fail, and paste any failure. If the PR adds tests, check they actually test the KV
   cache equivalence rather than just calling it.

3. **Benchmark the speed claim.** It's a perf PR, so measure it rather than believing it: time
   training steps/sec and inference latency on main versus the PR branch on this box, same
   settings, and report both numbers. If the speedup isn't real, say so with the measurements.

4. **Comment on the PR** with `beckett gh` — inline comments on the specific lines where you found
   something, plus one summary review comment. Be direct and technical, and be fair: if it's good
   work, say that. No swearing, no jokes — this is a public PR on someone else's contribution and
   it's a paper trail.

5. **Merge if it holds up.** Green tests, checkpoint loads clean, cached and uncached decode match,
   no regression to the recent training-signal work, speedup real. If all that is true, merge it.
   If any of it fails, don't merge — leave the review, mark clearly what needs to change.

## Constraints

- Use `beckett gh` for everything on GitHub. Never raw `gh`, never raw `git push`.
- **Do not restart or redeploy the live `babble-bot` / `babble-train` units.** Review and merge
  only; going live is a separate decision.
- **Do not point freebuff or any ad-funded / free-tier coding tool at this repo.** babble is a
  real project and ro's standing rule is that those tools only touch throwaway repos.
- Don't rewrite the contributor's PR. If it needs changes, that's review feedback, not you
  silently reimplementing it. Small obvious fixes are fine if you say what you changed.

## Done means

- The diff has been read in full and the four risk areas above are each explicitly answered.
- Test results reported with real numbers, and the before/after benchmark reported with real
  numbers.
- Inline comments plus a summary review posted on PR #7.
- Either the PR is merged, or there's a clear statement of exactly what blocks it.

## Checklist
- [ ] (worker fills this in as its FIRST action: concrete, verifiable items)

## Notes
(worker scratch: decisions, blockers, handoff notes)
