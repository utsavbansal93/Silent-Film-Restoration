#!/usr/bin/env bash
# Fetch the Wikimedia Commons WebM of Lanka Dahan (1917).
# Idempotent: skips download if hash matches.

set -euo pipefail

URL="https://upload.wikimedia.org/wikipedia/commons/8/89/Lanka_Dahan_%281917%29_by_Dadasaheb_Phalke.webm"
# Expected: ~212 MB, SHA1 676e6bf7cc5a3edab44cbc83c41e4676bda72f71 (from Wikimedia API)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/source"
DEST="${DEST_DIR}/lanka_dahan_1917.webm"

mkdir -p "${DEST_DIR}"

if [ -f "${DEST}" ]; then
    SIZE=$(stat -f%z "${DEST}" 2>/dev/null || stat -c%s "${DEST}")
    # Wikimedia file is ~202 MB. If it looks right, skip.
    if [ "${SIZE}" -gt 100000000 ]; then
        echo "Source already present at ${DEST} (${SIZE} bytes). Skipping download."
        exit 0
    fi
fi

echo "Downloading from Wikimedia Commons..."
curl -L --fail --progress-bar -A "silent-film-restoration/0.1 (utsavbansal93@gmail.com)" \
    -o "${DEST}.tmp" "${URL}"
mv "${DEST}.tmp" "${DEST}"

SHA=$(shasum -a 256 "${DEST}" | awk '{print $1}')
echo "Downloaded: ${DEST}"
echo "sha256: ${SHA}"
echo "${SHA}  $(basename "${DEST}")" > "${DEST}.sha256"
