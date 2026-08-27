#!/usr/bin/env bash
# Install the project git hooks into .git/hooks/.
#
# Safe to run repeatedly. Run this once after a fresh clone:
#   ./scripts/install-hooks.sh
#
# Installed hooks (both tracked symlinks so they stay under version
# control without `git config core.hooksPath`):
#   .git/hooks/pre-commit -> ../../scripts/pre-commit       (fast: ruff + black + public-API per commit)
#   .git/hooks/pre-push    -> ../../scripts/pre-commit-full (CI-parity: + compileall + pytest, per push)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

HOOKS_DIR=".git/hooks"
mkdir -p "$HOOKS_DIR"

install_hook() {
    local target="$1" source="$2"
    if [[ -L "$target" ]]; then
        rm "$target"
        echo "[install-hooks] removed existing symlink at $target"
    fi
    ln -s "$source" "$target"
    echo "[install-hooks] installed $target -> $source"
}

install_hook "$HOOKS_DIR/pre-commit" "../../scripts/pre-commit"
install_hook "$HOOKS_DIR/pre-push" "../../scripts/pre-commit-full"

chmod +x scripts/pre-commit scripts/pre-commit-full
echo "[install-hooks] done. test with: scripts/pre-commit ; scripts/pre-commit-full"
