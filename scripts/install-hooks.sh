#!/usr/bin/env bash
# Install the project pre-commit hook into .git/hooks/.
#
# Safe to run repeatedly. Run this once after a fresh clone:
#   ./scripts/install-hooks.sh
#
# The hook script lives in scripts/pre-commit (tracked in git) and is
# symlinked into .git/hooks/pre-commit. This keeps the hook under
# version control without `git config core.hooksPath`.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

HOOKS_DIR=".git/hooks"
TARGET="$HOOKS_DIR/pre-commit"
SOURCE="../../scripts/pre-commit"

mkdir -p "$HOOKS_DIR"

if [[ -L "$TARGET" ]]; then
    rm "$TARGET"
    echo "[install-hooks] removed existing symlink at $TARGET"
fi

ln -s "$SOURCE" "$TARGET"
chmod +x scripts/pre-commit
echo "[install-hooks] installed pre-commit hook -> $TARGET"
echo "[install-hooks] test with:  scripts/pre-commit"
