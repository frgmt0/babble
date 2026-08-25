# CLAUDE.md — working on babble / booper

babble is a Discord bot ("booper") that collects a consented corpus from people
talking to it and serves a small local language model. The README is the
authoritative deep documentation — read the relevant section before changing
consent, training, serving, or publishing behavior. This file is the
operational map: where things run, what must never be broken, and how to work
without stepping on the live bot or each other.

## The golden rule: dev tree vs live install

There are always (at least) two checkouts of this repo on a deploy box:

| | path | purpose |
| --- | --- | --- |
| dev | `~/projects/babble` (this checkout) | branches, edits, tests, PRs |
| live | `~/babble-live` | plain clone of `main`, runs the bot as systemd unit `babble-bot` |

- **Never edit code in `~/babble-live`.** It must stay a clean, unmodified
  clone of `origin/main` so `deploy/update-live.sh` can `merge --ff-only` it.
  All code changes happen in the dev tree and land via `main`.
- **Never run `babble fake-data`, tests, or experiments with the live dirs as
  cwd or `BABBLE_DATA_DIR`.** The live `data/` holds real people's consented
  messages, consent records, and the hash salt. Fake rows in there look like
  things real people said; deleting `data/.salt` breaks `!babble forget`
  permanently. Tests and fakes belong in the dev tree, which keeps no real
  data.
- Secrets live only in `~/babble-live/.env` (Discord token, webhook URL,
  HF_TOKEN). Never commit a `.env`, never paste its values into code, docs,
  logs, or PRs.

## Live deployments

- **This box (jason)**: `~/babble-live`, systemd `--user` unit `babble-bot`
  (linger enabled, so it survives logout/reboot). Bot account: `booper#9024`.
- **beckett's box**: `/home/beckett/babble-live` — the deployment the repo's
  promote/regression reports (`docs/reports/PRETRAIN_PROMOTE_LIVE.md` etc.) describe. Paths
  in those docs refer to that machine, not this one.

Only one bot process may hold a given Discord token. Don't start a second
`babble bot` (or run the unit on two boxes with the same token) — Discord will
bounce the sessions against each other.

## The models

Two HuggingFace repos hold our promoted checkpoints. They live under the
**ProCreations** account but they are this project's models:

- `ProCreations/booper-pretrain` — 34M-param base, pretrained on
  Ultra-FineWeb-L1 via `pretrain_hf.py` (BPE tokenizer, vocab 16384).
- `ProCreations/booper-chat` — the same model SFT'd on
  `mookiezi/Discord-Dialogues` in the **pair** layout
  (`<bos> prompt <sep> response <eos>`).

Local copies are kept in the live tree so serving never depends on the Hub:
`~/babble-live/artifacts/hf-booper-pretrain/` and `…/hf-booper-chat/`.

Checkpoint layout in `~/babble-live/checkpoints/`:

| file | is | role |
| --- | --- | --- |
| `latest.pt` | booper-chat | what the bot serves |
| `pretrained.pt` | booper-pretrain | stage-2 (post-train) base snapshot |
| `tokenizer.json` | BPE sidecar | shared — identical file in both HF repos |

## Invariants that have already caused live incidents

- **`BABBLE_SERVE_LAYOUT` must match the served checkpoint.** booper-chat is
  pair-layout, so the live `.env` sets `BABBLE_SERVE_LAYOUT=pair` (and
  `BABBLE_POST_LAYOUT=pair`). The checkpoint file carries no layout flag —
  nothing auto-detects this. Serve a pair checkpoint as `continuation` and it
  continues the user's message instead of answering; the reverse asks a
  pretrain checkpoint about a `<sep>` it never saw.
- **`babble train` starts from RANDOM INIT and overwrites `latest.pt`.** With
  a promoted HF checkpoint live, the corpus-growth auto-trigger would clobber
  a 34M model with a from-scratch 3.3M one. The live `.env` therefore sets
  `BABBLE_TRAIN_TRIGGER_ROWS=0` and `BABBLE_POST_TRIGGER_PAIRS=0` (manual
  only). Do not remove those lines casually; re-enabling auto-train is a
  decision about *which model booper is*, not a config tweak.
- **A checkpoint and its `tokenizer.json` travel together.** Serving refuses a
  vocab mismatch; a missing sidecar with vocab 16384 is a hard error (only the
  historical vocab-260 byte checkpoints may run sidecar-less). When promoting
  or rolling back, move/copy both files, and back up the outgoing pair first
  (convention: `~/babble-live/backups/<reason>-<date>/checkpoints/`).
- **Never change or lose the hash salt** (`data/.salt` / `BABBLE_HASH_SALT`) —
  `!babble forget` finds people's rows by re-deriving their hash.
- **Consent gates fail closed** and are checked at capture, training, and
  export. Any change touching `consent.py`, `SCOPE_BY_SOURCE`, or the stores
  needs the tests run and the README's consent section re-read first.

## Deploying a change

```bash
# from the dev tree: branch, PR, land on main. Then:
cd ~/babble-live && git pull --ff-only && systemctl --user restart babble-bot
tail -f logs/babble.log        # wait for bot.ready — do not walk away before it
```

`deploy/update-live.sh` (plus `babble-update.timer`) automates exactly this,
refuses dirty trees and in-flight training, and alerts on failure — prefer
wiring it up over ad-hoc pulls if updates become routine.

Promotion of *model weights* is separate from code deploys: copy the new
`latest.pt` + `tokenizer.json` into `checkpoints/` (backup first, see above),
set the matching `BABBLE_SERVE_LAYOUT`, restart, then verify with a real ping
and `babble logs`. Record what was promoted and the sha256s in a dated
markdown report in `docs/reports/` — that is the existing convention
(`docs/reports/PRETRAIN_PROMOTE_LIVE.md`).

## Quick reference (live box)

```bash
systemctl --user status babble-bot
~/babble-live/.venv/bin/babble logs --follow    # live event feed
~/babble-live/.venv/bin/babble summary          # rows, consent, checkpoint state
~/babble-live/.venv/bin/babble sample --prompt "hey"
systemctl --user restart babble-bot             # after any .env change
```

Dev setup: `uv venv && uv pip install -e ".[dev]"` (uv pulls CPU-only torch;
Python ≥3.10 — the live venv is 3.12). `pytest` runs 500+ tests, none of which
need a token or network.

## Coordinating with other Claude sessions

Claude Code supports inter-session messaging: `ListAgents` shows other live
sessions (local sessions on this machine, cloud sessions, and subagents you
spawned), and `SendMessage` delivers a message to one by name. Several people
— and several Claude sessions — work on this project at once. **Check before
you start, not after you collide:**

1. **At the start of any non-trivial task, run `ListAgents`.** If another
   session is listed, assume it may be mid-task in this repo. Message it
   (`SendMessage`) with one line: what you're about to touch (paths/areas) and
   ask what it's holding. Note: cloud sessions can read your message but
   cannot reply back into your session — check their own transcript, or just
   avoid the areas they were started for.
2. **Also check the tree itself** — sessions and humans both leave tracks:
   `git status` + `git branch --show-current` before you begin. A dirty tree
   or an unexpected branch means someone's mid-work: do not switch branches,
   stash, reset, or "clean up" their state. Ask (the user, or the session via
   SendMessage) first.
3. **One working tree, one branch.** Sessions sharing `~/projects/babble`
   share its checked-out branch — switching it swaps files under the other
   session's feet. For genuinely parallel work, make a separate `git worktree`
   (or spawn subagents with worktree isolation) instead of branch-flipping the
   shared checkout.
4. **Claim by scope, keep scopes disjoint.** When fanning out subagents or
   splitting with another session, give each worker an explicit file/dir scope
   with exactly one owner per file. "You own all *.md edits repo-wide; I own
   whole-file keep/delete under experiments/" cannot conflict; two workers
   "each fixing references" will.
5. **The live box is single-writer.** Only one session at a time may restart
   `babble-bot`, pull `~/babble-live`, promote checkpoints, or edit the live
   `.env`. If you didn't start that operation and can't confirm nobody else
   is mid-deploy, don't touch it.
6. **Before ending a work session**, leave the tree in a state others can walk
   into: committed (on a branch) or cleanly described to the user — not a pile
   of unexplained working-tree changes.

## Things Claude should not do unprompted

- Start/stop/restart `babble-bot`, promote or roll back checkpoints, or edit
  the live `.env` — propose it, with the exact commands, unless the user asked.
- Push datasets or models to HuggingFace, or set `HF_TOKEN` anywhere.
- Loosen anything in the consent, blocklist, or hashing paths.
- Commit generated data (`data/`, `checkpoints/`, `logs/`, `artifacts/*.pt`)
  or any `.env`.
