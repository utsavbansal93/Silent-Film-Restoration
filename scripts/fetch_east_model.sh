#!/usr/bin/env bash
# Fetch the pretrained EAST text detection model for S01 intertitle detection.
# ~94 MB. Idempotent: skips if already present with a non-zero size.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/models"
DEST="${DEST_DIR}/frozen_east_text_detection.pb"

# Primary URL (OpenCV community mirror). If this 404s, fall back to archived copy.
PRIMARY_URL="https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"

mkdir -p "${DEST_DIR}"

if [ -f "${DEST}" ]; then
    SIZE=$(stat -f%z "${DEST}" 2>/dev/null || stat -c%s "${DEST}")
    if [ "${SIZE}" -gt 50000000 ]; then
        echo "EAST model already present at ${DEST}. Skipping."
        exit 0
    fi
fi

echo "Downloading EAST model..."
curl -L --fail --progress-bar -o "${DEST}.tmp" "${PRIMARY_URL}"
mv "${DEST}.tmp" "${DEST}"

SHA=$(shasum -a 256 "${DEST}" | awk '{print $1}')
echo "EAST model: ${DEST}"
echo "sha256: ${SHA}"
