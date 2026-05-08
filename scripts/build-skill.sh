#!/usr/bin/env bash
# build-skill.sh — package this repo as a claude.ai-upload-ready .skill file.
# Usage: bash scripts/build-skill.sh  (run from repo root)
#
# Produces dist/watch.skill, a zip with a single top-level `watch/` directory
# containing SKILL.md and the scripts/ runtime. claude.ai's skill upload has a
# 200-file cap; `export-ignore` in .gitattributes + the zip -d strips below
# keep the bundle lean.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree is dirty; commit or stash before building" >&2
  exit 1
fi

# Frontmatter format check: YAML inline-list syntax `allowed-tools: [a, b]`
# silently breaks slash-command registration in Claude Code (upstream issue
# #6 — bradautomates/claude-video). The supported form is a bare
# comma-separated string. Bail if any frontmatter regresses to the
# bracketed form before we package the skill.
BAD_FRONTMATTER=$(grep -lE '^allowed-tools:[[:space:]]*\[' SKILL.md commands/*.md 2>/dev/null || true)
if [ -n "$BAD_FRONTMATTER" ]; then
  echo "error: YAML inline-list syntax in allowed-tools (breaks slash-command registration):" >&2
  echo "$BAD_FRONTMATTER" | sed 's/^/  /' >&2
  echo "       fix: change \`allowed-tools: [a, b, c]\` to \`allowed-tools: a, b, c\`" >&2
  exit 1
fi

mkdir -p dist
OUT="dist/watch.skill"
git archive --format=zip --prefix=watch/ --output="$OUT" HEAD

# claude.ai's .skill bundle needs only SKILL.md + scripts/ runtime. Claude Code
# needs hooks/, commands/, and .claude-plugin/ in the git archive (that's why
# they are NOT in .gitattributes export-ignore), but the .skill bundle should
# strip them to keep a single canonical SKILL.md and stay well under the
# 200-file cap.
zip -d "$OUT" \
  "watch/hooks/*" \
  "watch/commands/*" \
  "watch/.claude-plugin/*" \
  > /dev/null 2>&1 || true

COUNT=$(unzip -l "$OUT" | tail -1 | awk '{print $2}')
SIZE=$(du -h "$OUT" | cut -f1)

if [ "$COUNT" -gt 200 ]; then
  echo "error: $COUNT files in zip, claude.ai's cap is 200" >&2
  echo "       check .gitattributes export-ignore entries and this script's zip -d excludes" >&2
  exit 1
fi

SKILL_MD_COUNT=$(unzip -l "$OUT" | grep -c "SKILL.md" || true)
if [ "$SKILL_MD_COUNT" -ne 1 ]; then
  echo "error: expected exactly one SKILL.md, found $SKILL_MD_COUNT" >&2
  exit 1
fi

echo "built $OUT ($COUNT files, $SIZE)"
echo "upload via the claude.ai skill UI"
