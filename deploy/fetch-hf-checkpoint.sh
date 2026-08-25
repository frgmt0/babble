#!/usr/bin/env bash
# Fetch a promoted checkpoint pair -- latest.pt + tokenizer.json -- from a
# HuggingFace model repo into a local checkpoints directory.
#
#   deploy/fetch-hf-checkpoint.sh ProCreations/boopit-1 ~/babble-boopit/checkpoints
#
# The two files travel together (see CLAUDE.md: serving refuses a vocab
# mismatch), so both are downloaded first and only then moved into place --
# a failed download can never leave the destination holding a mismatched pair.
# If the destination already holds a pair, back it up first, same convention
# as any promotion.
set -euo pipefail

repo="${1:?usage: fetch-hf-checkpoint.sh <namespace/repo> <dest-dir>}"
dest="${2:?usage: fetch-hf-checkpoint.sh <namespace/repo> <dest-dir>}"
base="https://huggingface.co/${repo}/resolve/main"

mkdir -p "$dest"
tmp="$(mktemp -d "${dest}/.fetch.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

for f in latest.pt tokenizer.json; do
  echo "fetching ${repo}/${f} ..."
  if ! curl -fSL --retry 3 -o "${tmp}/${f}" "${base}/${f}"; then
    echo "error: could not fetch ${base}/${f}" >&2
    echo "       (repo missing, private, or the file has not been uploaded yet)" >&2
    exit 1
  fi
done

for f in latest.pt tokenizer.json; do
  mv "${tmp}/${f}" "${dest}/${f}"
done

echo "fetched into ${dest}:"
sha256sum "${dest}/latest.pt" "${dest}/tokenizer.json"
