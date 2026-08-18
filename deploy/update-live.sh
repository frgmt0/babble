#!/usr/bin/env bash
# Keep a plain-clone live install current with origin/main, safely.
#
# Idempotent and meant to run on a timer (see babble-update.timer/.service in
# this directory): fetches, and only touches anything if it is actually
# behind. Every decision -- including "did nothing" -- is written to the same
# babble.log/babble.jsonl the bot itself writes, so drift is auditable with
# `babble logs`. See the README's "Keeping the live install current" section.
#
# Refuses, loudly (non-zero exit), to:
#   - touch a repo whose 'origin' isn't actually kowo-co/babble (never
#     auto-repoints it -- a wrong remote should be noisy, not "fixed")
#   - merge over uncommitted tracked changes
#   - merge, rebase or force anything -- only `git merge --ff-only`
#   - report success if the bot doesn't come back up after a restart
#
# Skips (exit 0, try again next tick) rather than fails when a training run
# is currently in flight, so a restart never kills one mid-write.
#
# Every path below is overridable by env var so this runs from any checkout,
# not just this box:
#   BABBLE_LIVE_DIR             the live checkout (default: $HOME/babble-live)
#   BABBLE_UPDATE_REMOTE        the origin url that must be configured
#                                (default: https://github.com/kowo-co/babble.git)
#   BABBLE_UPDATE_BRANCH        branch to track (default: main)
#   BABBLE_BOT_UNIT              systemd --user unit name (default: babble-bot)
#   BABBLE_DATA_DIR              where update_state.json is written
#                                (default: $BABBLE_LIVE_DIR/data)
#   BABBLE_LOG_DIR                where babble.log / babble.jsonl live
#                                (default: $BABBLE_LIVE_DIR/logs)
#   BABBLE_UPDATE_VENV_BIN        directory containing `uv` if not on PATH
#   BABBLE_UPDATE_RESTART_TIMEOUT  seconds to wait for bot.ready (default: 90)
#   BABBLE_UPDATE_POLL_INTERVAL    seconds between readiness polls (default: 2)
#   BABBLE_TRAIN_SUBCOMMANDS       space-separated `babble` subcommands that
#                                count as "training in flight"
#                                (default: "train post-train")
set -euo pipefail

LIVE_DIR="${BABBLE_LIVE_DIR:-$HOME/babble-live}"
EXPECTED_REMOTE="${BABBLE_UPDATE_REMOTE:-https://github.com/kowo-co/babble.git}"
BRANCH="${BABBLE_UPDATE_BRANCH:-main}"
BOT_UNIT="${BABBLE_BOT_UNIT:-babble-bot}"
DATA_DIR="${BABBLE_DATA_DIR:-$LIVE_DIR/data}"
LOG_DIR="${BABBLE_LOG_DIR:-$LIVE_DIR/logs}"
RESTART_TIMEOUT="${BABBLE_UPDATE_RESTART_TIMEOUT:-90}"
POLL_INTERVAL="${BABBLE_UPDATE_POLL_INTERVAL:-2}"
TRAIN_SUBCOMMANDS="${BABBLE_TRAIN_SUBCOMMANDS:-train post-train}"

if [ -n "${BABBLE_UPDATE_VENV_BIN:-}" ]; then
  PATH="$BABBLE_UPDATE_VENV_BIN:$PATH"
fi

STATE_FILE="$DATA_DIR/update_state.json"
LOG_JSONL="$LOG_DIR/babble.jsonl"
LOG_TEXT="$LOG_DIR/babble.log"

mkdir -p "$DATA_DIR" "$LOG_DIR"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%S+00:00"; }

json_escape() {
  local s=$1
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  printf '%s' "$s"
}

# Appends one line to babble.jsonl and one to babble.log, the same files
# `babble logs` reads -- so a run of this script shows up next to bot and
# training events instead of living in a log nobody looks at.
log_event() {
  local event=$1; shift
  local ts; ts=$(now_iso)
  local json="{\"ts\":\"$ts\",\"component\":\"update\",\"event\":\"$event\""
  local human="$ts  update.$event"
  local kv key val
  for kv in "$@"; do
    key=${kv%%=*}
    val=${kv#*=}
    json="$json,\"$key\":\"$(json_escape "$val")\""
    human="$human $key=$val"
  done
  json="$json}"
  printf '%s\n' "$json" >>"$LOG_JSONL"
  printf '%s\n' "$human" >>"$LOG_TEXT"
}

fail() {
  log_event "failed" "reason=$1"
  echo "update-live: $1" >&2
  exit 1
}

# $1 local sha  $2 remote sha  $3 up_to_date (true/false)  $4 last_action
write_state() {
  local tmp="$STATE_FILE.tmp.$$"
  cat >"$tmp" <<EOF
{
  "checked_at": "$(now_iso)",
  "local_commit": "$1",
  "remote_commit": "$2",
  "up_to_date": $3,
  "last_action": "$4"
}
EOF
  mv "$tmp" "$STATE_FILE"
}

# True if a `babble train` process is currently running, from any user's
# cmdline on the box. Reads each process's real argv out of /proc
# (NUL-separated, so a token is compared to "train" exactly) rather than
# grepping the flattened command line with `pgrep -f` -- a substring match
# there is one long unrelated argument containing the words "babble" and
# "train" away from a false positive (an editor buffer, a log tail, another
# agent's prompt on a shared box).
#
# Two shapes both count: the auto-trigger's own `python -m babble train`
# (trainer.py's AutoTrainTrigger), and the installed console script, whose
# shebang the kernel resolves to `<venv>/python <venv>/bin/babble train`
# -- argv0 is the interpreter in both, so the second token is what tells them
# apart from an unrelated python process.
training_in_flight() {
  local pid args base0 base1 i is_babble start sub
  for pid in /proc/[0-9]*; do
    [ -r "$pid/cmdline" ] || continue
    args=()
    while IFS= read -r -d '' a; do args+=("$a"); done <"$pid/cmdline" 2>/dev/null
    [ "${#args[@]}" -ge 2 ] || continue
    base0=${args[0]##*/}
    is_babble=0
    start=0
    if [ "$base0" = "babble" ]; then
      is_babble=1
      start=1
    elif [[ "$base0" == python* ]]; then
      if [ "${args[1]}" = "-m" ] && [ "${#args[@]}" -ge 3 ] && [ "${args[2]}" = "babble" ]; then
        is_babble=1
        start=3
      else
        base1=${args[1]##*/}
        if [ "$base1" = "babble" ]; then
          is_babble=1
          start=2
        fi
      fi
    fi
    [ "$is_babble" = 1 ] || continue
    for ((i = start; i < ${#args[@]}; i++)); do
      for sub in $TRAIN_SUBCOMMANDS; do
        [ "${args[$i]}" = "$sub" ] && return 0
      done
    done
  done
  return 1
}

# Accepts https://github.com/org/repo(.git) and git@github.com:org/repo(.git)
# as the same remote, so either clone style passes.
normalize_remote() {
  local u=$1
  u=${u%.git}
  u=${u#git@github.com:}
  u=${u#https://github.com/}
  u=${u#http://github.com/}
  u=${u,,}
  printf '%s' "$u"
}

[ -d "$LIVE_DIR" ] || fail "BABBLE_LIVE_DIR=$LIVE_DIR does not exist"
cd "$LIVE_DIR"
git rev-parse --git-dir >/dev/null 2>&1 || fail "$LIVE_DIR is not a git checkout"

# --- 1. the origin must actually be us. Never repointed automatically: a
# wrong remote silently means "will never pull anything real", which is
# exactly the 2026-08-15 bug this script exists to catch.
origin_url=$(git remote get-url origin 2>/dev/null || true)
[ -n "$origin_url" ] || fail "no 'origin' remote is configured in $LIVE_DIR"
if [ "$(normalize_remote "$origin_url")" != "$(normalize_remote "$EXPECTED_REMOTE")" ]; then
  fail "origin is '$origin_url', not $EXPECTED_REMOTE -- refusing to touch it (not auto-correcting; fix the remote by hand)"
fi

# --- 2. fetch, then compare.
git fetch --quiet origin "$BRANCH" || fail "git fetch origin $BRANCH failed"

local_sha=$(git rev-parse HEAD)
remote_sha=$(git rev-parse "origin/$BRANCH")

if [ "$local_sha" = "$remote_sha" ]; then
  write_state "$local_sha" "$remote_sha" true "noop"
  log_event "noop" "commit=${local_sha:0:12}"
  exit 0
fi

# --- 3. refuse to merge over uncommitted tracked changes.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  write_state "$local_sha" "$remote_sha" false "skipped_dirty"
  fail "working tree in $LIVE_DIR has uncommitted tracked changes -- refusing to merge"
fi

# --- 4. never restart mid-write: a training run in flight means skip this
# cycle, not fail it -- the next tick will pick it up once the run has
# finished and written its checkpoint.
if training_in_flight; then
  write_state "$local_sha" "$remote_sha" false "skipped_training"
  log_event "skipped" "reason=training_in_flight" "commit=${local_sha:0:12}"
  exit 0
fi

# --- 5. fast-forward only. Never merge, never rebase, never force.
git merge --ff-only --quiet "origin/$BRANCH" || {
  write_state "$local_sha" "$remote_sha" false "failed_merge"
  fail "git merge --ff-only origin/$BRANCH failed (local history has diverged?)"
}
new_sha=$(git rev-parse HEAD)
log_event "merged" "from=${local_sha:0:12}" "to=${new_sha:0:12}"

# --- 6. sync dependencies, only if the lockfile actually moved.
if git diff --name-only "$local_sha" "$new_sha" | grep -qx 'uv.lock'; then
  if command -v uv >/dev/null 2>&1; then
    if uv sync --quiet; then
      log_event "synced" "commit=${new_sha:0:12}"
    else
      write_state "$new_sha" "$remote_sha" false "failed_sync"
      fail "uv sync failed after merging to ${new_sha:0:12}"
    fi
  else
    log_event "sync_skipped" "reason=uv_not_found"
  fi
fi

# --- 7. restart, then prove it actually came back rather than trusting the
# exit code of `restart` (which only means "systemd accepted the request").
offset_before=0
[ -f "$LOG_TEXT" ] && offset_before=$(wc -c <"$LOG_TEXT")

systemctl --user restart "$BOT_UNIT" || {
  write_state "$new_sha" "$remote_sha" false "failed_restart"
  fail "systemctl --user restart $BOT_UNIT failed"
}
log_event "restarted" "unit=$BOT_UNIT" "commit=${new_sha:0:12}"

deadline=$(( $(date +%s) + RESTART_TIMEOUT ))
ready=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ -f "$LOG_TEXT" ] && tail -c "+$((offset_before + 1))" "$LOG_TEXT" 2>/dev/null | grep -q 'bot\.ready'; then
    ready=1
    break
  fi
  if ! systemctl --user is-active --quiet "$BOT_UNIT"; then
    break
  fi
  sleep "$POLL_INTERVAL"
done

if [ -z "$ready" ]; then
  write_state "$new_sha" "$remote_sha" false "failed_restart_verify"
  fail "$BOT_UNIT did not log bot.ready within ${RESTART_TIMEOUT}s of restart"
fi

write_state "$new_sha" "$remote_sha" true "updated"
log_event "done" "commit=${new_sha:0:12}"
