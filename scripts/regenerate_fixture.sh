#!/usr/bin/env bash
# Carve the 30-second test fixture from the full source.
#
# Initial behaviour: first 30 seconds (good enough for S00 development).
# After S01 has run once on the full source and produced probe_report.json with
# per-frame motion data, update FIXTURE_START_SECS (and optionally FIXTURE_SHOT_HINT)
# to carve the motion-heavy slice instead. See JOURNAL.md decision D6.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${REPO_ROOT}/source/lanka_dahan_1917.webm"
DEST_DIR="${REPO_ROOT}/tests/fixtures"
DEST="${DEST_DIR}/test_clip_30sec.webm"

# Carve window. Change these after S01 data is available.
FIXTURE_START_SECS="${FIXTURE_START_SECS:-0}"
FIXTURE_DURATION_SECS="${FIXTURE_DURATION_SECS:-30}"

mkdir -p "${DEST_DIR}"

if [ ! -f "${SOURCE}" ]; then
    echo "Source missing: ${SOURCE}"
    echo "Run: just fetch-source"
    exit 1
fi

echo "Carving ${FIXTURE_DURATION_SECS}s fixture starting at ${FIXTURE_START_SECS}s..."
ffmpeg -y -hide_banner -loglevel warning \
    -ss "${FIXTURE_START_SECS}" \
    -i "${SOURCE}" \
    -t "${FIXTURE_DURATION_SECS}" \
    -c:v copy -an \
    "${DEST}"

SHA=$(shasum -a 256 "${DEST}" | awk '{print $1}')
echo "Fixture: ${DEST}"
echo "sha256: ${SHA}"
