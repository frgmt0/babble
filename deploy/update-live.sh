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
# A refusal or failure is never just a failed systemd unit: `data/update_state.json`
# (already the record of the last check) also carries a `consecutive_failures`
# counter and a `commits_behind` count, and every such run appends an
# `update.alert` log line and, once (first failure) then every
# BABBLE_UPDATE_ALERT_EVERY_N-th failure after that, both drops a
# `data/UPDATE_FAILING` marker file and -- if BABBLE_LOG_WEBHOOK_URL (the same
# webhook the training feed uses, see .env.example) is set -- posts one line
# to it. Unconfigured webhook is silent, same convention as the rest of the
# repo's Discord feed. The marker and counter both clear on the next clean
# run, so a resolved drift does not lie forever.
#
# `--check` prints a one-line, network-free status (clean/dirty, commits
# behind *and* ahead of origin/main, last update action, consecutive
# failures) and exits 0 if the tree is clean *and* not diverged, or 1 if
# dirty or if HEAD has commits that are not on origin -- so this is
# greppable in one command instead of requiring a `git status` on the box.
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
#   BABBLE_UPDATE_ALERT_EVERY_N    alert on the 1st failure, then every Nth
#                                consecutive failure after that (default: 5)
#   BABBLE_LOG_WEBHOOK_URL        optional Discord webhook for alerts (shared
#                                with the training feed; unset = no post)
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
ALERT_EVERY_N="${BABBLE_UPDATE_ALERT_EVERY_N:-5}"
[ "$ALERT_EVERY_N" -gt 0 ] 2>/dev/null || ALERT_EVERY_N=5
TRAIN_SUBCOMMANDS="${BABBLE_TRAIN_SUBCOMMANDS:-train post-train}"

if [ -n "${BABBLE_UPDATE_VENV_BIN:-}" ]; then
  PATH="$BABBLE_UPDATE_VENV_BIN:$PATH"
fi

STATE_FILE="$DATA_DIR/update_state.json"
MARKER_FILE="$DATA_DIR/UPDATE_FAILING"
LOG_JSONL="$LOG_DIR/babble.jsonl"
LOG_TEXT="$LOG_DIR/babble.log"

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
# $5 consecutive_failures  $6 commits_behind (a number, or the literal null)
write_state() {
  local tmp="$STATE_FILE.tmp.$$"
  cat >"$tmp" <<EOF
{
  "checked_at": "$(now_iso)",
  "local_commit": "$1",
  "remote_commit": "$2",
  "up_to_date": $3,
  "last_action": "$4",
  "consecutive_failures": $5,
  "commits_behind": $6
}
EOF
  mv "$tmp" "$STATE_FILE"
}

# Pulls one integer/string field back out of the last-written state file, so
# a fresh run knows how many consecutive failures came before it without a
# second state file. Plain grep/sed on purpose -- update_state.json is a
# small, fixed shape this script itself writes, and pulling in `jq` (or
# python, which the live venv does not even carry numpy under) for one field
# is not worth a new dependency.
state_field_int() {
  local key=$1 default=$2 val=""
  if [ -f "$STATE_FILE" ]; then
    val=$(grep -o "\"$key\"[[:space:]]*:[[:space:]]*[0-9]*" "$STATE_FILE" 2>/dev/null | grep -o '[0-9]*$' | head -1) || true
  fi
  printf '%s' "${val:-$default}"
}

state_field_str() {
  local key=$1 default=$2 val=""
  if [ -f "$STATE_FILE" ]; then
    val=$(grep -o "\"$key\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$STATE_FILE" 2>/dev/null | sed -E 's/.*: *"([^"]*)"$/\1/' | head -1) || true
  fi
  printf '%s' "${val:-$default}"
}

# How many commits HEAD is behind $2, or the literal `null` (unknown) when
# $2 isn't resolvable -- e.g. a fetch just failed and origin/$BRANCH has
# never been fetched at all yet.
commits_behind() {
  local loc=$1 rem=$2 n
  if [ -z "$rem" ]; then
    printf 'null'
    return
  fi
  n=$(git rev-list --count "$loc..$rem" 2>/dev/null) || n=""
  printf '%s' "${n:-null}"
}

commits_ahead() {
  local loc=$1 rem=$2 n
  if [ -z "$rem" ]; then
    printf 'null'
    return
  fi
  n=$(git rev-list --count "$rem..$loc" 2>/dev/null) || n=""
  printf '%s' "${n:-null}"
}

# First failure, then every ALERT_EVERY_N-th one after that -- the same
# "first-then-every-fifth" shape as a rate-limited alarm, so a stuck update
# is impossible to miss but does not spam every 5 minutes forever. Always
# logs; only drops the marker file / posts to Discord on an actual alert
# tick, and the marker's content differs every time (growing count and
# behind-number) rather than repeating byte-for-byte.
maybe_alert() {
  local reason=$1 count=$2 behind=$3 behind_display=$3
  [ "$behind_display" = "null" ] && behind_display="unknown"
  log_event "alert" "reason=$reason" "consecutive_failures=$count" "commits_behind=$behind_display"
  if [ "$count" -ne 1 ] && [ $((count % ALERT_EVERY_N)) -ne 0 ]; then
    return
  fi
  local msg
  msg="update-live ALERT: $reason -- $LIVE_DIR is $count consecutive checks behind (commits_behind=$behind_display over origin/$BRANCH)"
  {
    printf '%s\n' "$msg"
    printf 'checked_at: %s\n' "$(now_iso)"
  } >"$MARKER_FILE"
  if [ -n "${BABBLE_LOG_WEBHOOK_URL:-}" ] && command -v curl >/dev/null 2>&1; then
    curl -fsS -m 5 -X POST -H 'Content-Type: application/json' \
      -d "{\"content\":\"$(json_escape "$msg")\",\"allowed_mentions\":{\"parse\":[]}}" \
      "$BABBLE_LOG_WEBHOOK_URL" >/dev/null 2>&1 || true
  fi
}

# Clears the marker left by a prior failing streak once things are current
# again, so a resolved drift does not go on claiming to be broken.
clear_alert() {
  if [ -f "$MARKER_FILE" ]; then
    rm -f "$MARKER_FILE"
    log_event "recovered" "commit=${1:0:12}"
  fi
}

# Writes state with an incremented failure count, alerts, then fails loudly.
# $1 local sha  $2 remote sha  $3 last_action  $4 human-readable fail message
fail_drift() {
  local local_sha=$1 remote_sha=$2 action=$3 message=$4 prev count behind
  prev=$(state_field_int consecutive_failures 0)
  count=$((prev + 1))
  behind=$(commits_behind "$local_sha" "$remote_sha")
  write_state "$local_sha" "$remote_sha" false "$action" "$count" "$behind"
  maybe_alert "$action" "$count" "$behind"
  fail "$message"
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

# --- --check: a network-free, one-line status query. Reads whatever
# origin/$BRANCH ref the last real run already fetched (never fetches
# itself, same "never hit the network just to answer a question" rule
# `babble summary` follows) plus the state file's own bookkeeping, so it is
# safe to run as often as you like by hand.
if [ "${1:-}" = "--check" ]; then
  [ -d "$LIVE_DIR" ] || { echo "update-live --check: BABBLE_LIVE_DIR=$LIVE_DIR does not exist" >&2; exit 2; }
  cd "$LIVE_DIR"
  git rev-parse --git-dir >/dev/null 2>&1 || { echo "update-live --check: $LIVE_DIR is not a git checkout" >&2; exit 2; }
  status="clean"
  [ -n "$(git status --porcelain --untracked-files=no)" ] && status="dirty"
  behind="unknown"
  ahead="unknown"
  if git rev-parse "origin/$BRANCH" >/dev/null 2>&1; then
    behind=$(git rev-list --count "HEAD..origin/$BRANCH" 2>/dev/null) || behind="unknown"
    ahead=$(git rev-list --count "origin/$BRANCH..HEAD" 2>/dev/null) || ahead="unknown"
  fi
  last_action=$(state_field_str last_action unknown)
  consecutive=$(state_field_int consecutive_failures 0)
  checked_at=$(state_field_str checked_at never)
  printf 'status=%s behind=%s ahead=%s last_action=%s consecutive_failures=%s checked_at=%s\n' \
    "$status" "$behind" "$ahead" "$last_action" "$consecutive" "$checked_at"
  [ "$status" = "clean" ] || exit 1
  # Divergence is as loud as dirtiness: HEAD commits that origin does not
  # have cannot be fast-forwarded away, so --check must not look "fine".
  if [ "$ahead" != "unknown" ] && [ "$ahead" != "0" ]; then
    exit 1
  fi
  exit 0
fi

mkdir -p "$DATA_DIR" "$LOG_DIR"

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
local_sha=$(git rev-parse HEAD)
if ! git fetch --quiet origin "$BRANCH"; then
  remote_sha=$(git rev-parse "origin/$BRANCH" 2>/dev/null || printf '')
  fail_drift "$local_sha" "$remote_sha" "failed_fetch" "git fetch origin $BRANCH failed"
fi

remote_sha=$(git rev-parse "origin/$BRANCH")

if [ "$local_sha" = "$remote_sha" ]; then
  write_state "$local_sha" "$remote_sha" true "noop" 0 0
  clear_alert "$local_sha"
  log_event "noop" "commit=${local_sha:0:12}"
  exit 0
fi

# --- 3. refuse to merge over uncommitted tracked changes.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail_drift "$local_sha" "$remote_sha" "skipped_dirty" \
    "working tree in $LIVE_DIR has uncommitted tracked changes -- refusing to merge"
fi

# --- 4. never restart mid-write: a training run in flight means skip this
# cycle, not fail it -- the next tick will pick it up once the run has
# finished and written its checkpoint. Not a failure, so it neither
# increments nor resets the consecutive-failure count.
if training_in_flight; then
  prev_failures=$(state_field_int consecutive_failures 0)
  behind=$(commits_behind "$local_sha" "$remote_sha")
  write_state "$local_sha" "$remote_sha" false "skipped_training" "$prev_failures" "$behind"
  log_event "skipped" "reason=training_in_flight" "commit=${local_sha:0:12}"
  exit 0
fi

# --- 5. fast-forward only. Never merge, never rebase, never force.
# A checkout whose HEAD is not an ancestor of origin/$BRANCH cannot
# fast-forward -- the 2026-08-25 live-box incident. Treat that the same
# way as a dirty tree (own reason string, same fail_drift / first-then-
# every-Nth alert path) instead of letting `git merge --ff-only` fail
# with a generic "failed_merge".
if ! git merge-base --is-ancestor "$local_sha" "$remote_sha"; then
  ahead=$(commits_ahead "$local_sha" "$remote_sha")
  fail_drift "$local_sha" "$remote_sha" "skipped_diverged" \
    "git merge --ff-only origin/$BRANCH is not possible (local history has diverged; $ahead commit(s) on HEAD are not on origin/$BRANCH)"
fi
git merge --ff-only --quiet "origin/$BRANCH" || fail_drift "$local_sha" "$remote_sha" "failed_merge" \
  "git merge --ff-only origin/$BRANCH failed"
new_sha=$(git rev-parse HEAD)
log_event "merged" "from=${local_sha:0:12}" "to=${new_sha:0:12}"

# --- 6. sync dependencies, only if the lockfile actually moved.
if git diff --name-only "$local_sha" "$new_sha" | grep -qx 'uv.lock'; then
  if command -v uv >/dev/null 2>&1; then
    if uv sync --quiet; then
      log_event "synced" "commit=${new_sha:0:12}"
    else
      fail_drift "$new_sha" "$remote_sha" "failed_sync" "uv sync failed after merging to ${new_sha:0:12}"
    fi
  else
    log_event "sync_skipped" "reason=uv_not_found"
  fi
fi

# --- 7. restart, then prove it actually came back rather than trusting the
# exit code of `restart` (which only means "systemd accepted the request").
offset_before=0
[ -f "$LOG_TEXT" ] && offset_before=$(wc -c <"$LOG_TEXT")

systemctl --user restart "$BOT_UNIT" || fail_drift "$new_sha" "$remote_sha" "failed_restart" \
  "systemctl --user restart $BOT_UNIT failed"
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
  fail_drift "$new_sha" "$remote_sha" "failed_restart_verify" \
    "$BOT_UNIT did not log bot.ready within ${RESTART_TIMEOUT}s of restart"
fi

write_state "$new_sha" "$remote_sha" true "updated" 0 0
clear_alert "$new_sha"
log_event "done" "commit=${new_sha:0:12}"
