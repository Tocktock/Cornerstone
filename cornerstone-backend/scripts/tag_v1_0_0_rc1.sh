#!/usr/bin/env bash
set -euo pipefail

# Tag Cornerstone Backend v1.0.0-rc.1 from the already verified v0.13.1 commit.
# Run from the Git repository root.
#
# This script does NOT commit new code. It verifies that v0.13.1 points at HEAD,
# runs the release-candidate checker, scans for Notion token patterns, then creates
# annotated tag v1.0.0-rc.1.

BACKEND_DIR="cornerstone-backend"
V0131_TAG="v0.13.1"
RC_TAG="v1.0.0-rc.1"
EXPECTED_VERSION="0.13.1"

echo "== Cornerstone Backend ${RC_TAG} tag helper =="

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "ERROR: not inside a Git repository."
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ ! -d "$BACKEND_DIR" ]]; then
  echo "ERROR: expected directory '$BACKEND_DIR' not found at repo root."
  exit 1
fi

echo "Repository: $REPO_ROOT"
echo

echo "== Git state =="
git status --short

if [[ -n "$(git status --porcelain)" ]]; then
  echo
  echo "ERROR: worktree is not clean."
  echo "Commit, stash, or remove changes before tagging ${RC_TAG}."
  exit 1
fi

if ! git rev-parse "$V0131_TAG" >/dev/null 2>&1; then
  echo "ERROR: required tag ${V0131_TAG} does not exist."
  exit 1
fi

HEAD_SHA="$(git rev-parse HEAD)"
V0131_SHA="$(git rev-list -n 1 "$V0131_TAG")"

echo "HEAD:       $HEAD_SHA"
echo "${V0131_TAG}: $V0131_SHA"

if [[ "$HEAD_SHA" != "$V0131_SHA" ]]; then
  echo
  echo "ERROR: HEAD is not the same commit as ${V0131_TAG}."
  echo "Checkout the verified v0.13.1 commit before tagging ${RC_TAG}."
  exit 1
fi

if git rev-parse "$RC_TAG" >/dev/null 2>&1; then
  echo "ERROR: tag ${RC_TAG} already exists."
  exit 1
fi

echo
echo "== Version marker check =="
PYPROJECT_VERSION="$(python - <<'PY'
from pathlib import Path
import re
text = Path("cornerstone-backend/pyproject.toml").read_text()
m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
print(m.group(1) if m else "")
PY
)"
INIT_VERSION="$(python - <<'PY'
from pathlib import Path
import re
text = Path("cornerstone-backend/src/cornerstone/__init__.py").read_text()
m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
print(m.group(1) if m else "")
PY
)"
echo "pyproject.toml version: $PYPROJECT_VERSION"
echo "cornerstone.__version__: $INIT_VERSION"

if [[ "$PYPROJECT_VERSION" != "$EXPECTED_VERSION" || "$INIT_VERSION" != "$EXPECTED_VERSION" ]]; then
  echo "ERROR: expected both version markers to remain ${EXPECTED_VERSION} for RC tag."
  echo "RC tag should point to the verified v0.13.1 commit, not a new code version."
  exit 1
fi

echo
echo "== Release-candidate checker =="
(
  cd "$BACKEND_DIR"
  python scripts/check_release_candidate.py
)

echo
echo "== Token pattern scan =="
if command -v rg >/dev/null 2>&1; then
  if rg -n "[n]tn_" "$BACKEND_DIR/README.md" "$BACKEND_DIR/docs" "$BACKEND_DIR/scripts" "$BACKEND_DIR/src" "$BACKEND_DIR/tests" "$BACKEND_DIR/reports" "$BACKEND_DIR/pyproject.toml" "$BACKEND_DIR/.env.example"; then
    echo "ERROR: possible Notion token pattern found. Review before tagging."
    exit 1
  fi
  echo "No Notion token pattern found."
else
  echo "WARNING: ripgrep not installed; skipping token scan. Run manually:"
  echo "  rg -n \"[n]tn_\" cornerstone-backend/README.md cornerstone-backend/docs cornerstone-backend/scripts cornerstone-backend/src cornerstone-backend/tests cornerstone-backend/reports cornerstone-backend/pyproject.toml cornerstone-backend/.env.example"
fi

echo
echo "About to create annotated tag ${RC_TAG} on commit ${HEAD_SHA}."
read -r -p "Proceed? [y/N] " answer
if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

git tag -a "$RC_TAG" -m "cornerstone-backend ${RC_TAG}"
echo "Created tag: ${RC_TAG}"

echo
echo "Verify:"
echo "  git tag --points-at HEAD"
echo
echo "Push:"
echo "  git push origin ${V0131_TAG}"
echo "  git push origin ${RC_TAG}"
