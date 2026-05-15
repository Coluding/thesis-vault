#!/usr/bin/env bash
# vault-snapshot.sh — schedule via cron, takes a Git snapshot of the vault.
# This is a safety net, not a curated changelog. Don't read the diffs.
#
# Cron example (every 2 hours, only when vault has changes):
#   0 */2 * * * /path/to/thesis-vault/90_Meta/scripts/snapshot.sh >/dev/null 2>&1

set -euo pipefail

VAULT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$VAULT_DIR"

# Bail if not a git repo
if [ ! -d ".git" ]; then
  echo "Not a git repo. Run: git init && git remote add origin <your-repo>"
  exit 1
fi

# Bail if nothing changed
if [ -z "$(git status --porcelain)" ]; then
  exit 0
fi

git add -A
git commit -m "vault snapshot $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null

# Push if origin is configured. Don't fail the snapshot if push fails (offline, etc.)
if git remote get-url origin >/dev/null 2>&1; then
  git push origin HEAD >/dev/null 2>&1 || echo "Push failed — snapshot is local only."
fi
