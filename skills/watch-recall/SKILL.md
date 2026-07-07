---
name: watch-recall
version: "0.4.5"
description: Retrieve a previously saved /watch report by topic or title so this session can read, see, or learn from a video watched in an earlier session — without re-watching it. Full-text-searches the Paperless-ngx library (report transcript + OCR), falls back to matching Nextcloud folder titles, and points at the Immich album for frames.
argument-hint: "<topic-or-title>"
allowed-tools: Bash, Read
---

# /watch-recall

`/watch` saves every run's `report.md` (transcript + OCR + frame index) to a library — a Paperless-ngx document (full-text searchable), a Nextcloud folder, and an Immich album of frames (see the `watch` skill's "Library storage" section). This command reads that library back so a **later session** can learn from a video it never watched.

## Resolve `SKILL_DIR` and the script

Set `SKILL_DIR` to the absolute path of the directory containing THIS SKILL.md (your harness told you that path in the Read result). The retrieval script is a sibling of the `watch` skill's scripts:

```bash
SKILL_DIR="<absolute path of the directory containing the SKILL.md you Read>"
RECALL="$SKILL_DIR/../watch/scripts/recall.py"   # skills/watch/scripts/recall.py
if [ ! -f "$RECALL" ]; then
  echo "ERROR: recall.py not found at $RECALL" >&2
  exit 1
fi
```

On Windows substitute `python` for `python3`.

## How to invoke

**Step 1 — parse the input.** Everything after `/watch-recall` is the search query — a topic ("bevel modifier", "blender dice"), a title fragment, or a phrase you expect in the transcript.

**Step 2 — run the script.**

```bash
python3 "$RECALL" "<query>"
```

Flags:
- `--full` — print each matched report's *entire* text (transcript + OCR + frame index), not just a snippet. Use this when you actually want to learn the video's content, not just find it.
- `--limit N` — cap the number of reports returned (default 5).

**Step 3 — read the output and answer.** The script prints, per match: the title, the original video URL, where the full report lives (Paperless doc + Nextcloud path), and the Immich album name. In `--full` mode it also prints the whole `report.md`. Answer the user's question from the transcript/OCR text.

**Step 4 — pull frames only if you need to SEE something.** The report text is usually enough. When the user's question is genuinely visual ("what did the UI look like at that step?"), the frames are in the named **Immich album** — open it, or re-run the `watch` skill focused on that timestamp.

## Retrieval order (what the script does)

1. **Paperless-ngx full-text search** (`GET /api/documents/?query=…`) — finds reports by topic even when you don't know the title, then fetches each match's full `content`. Needs `PAPERLESS_TOKEN` in `~/.config/watch/.env`.
2. **Nextcloud folder-title match** (fallback) — when Paperless is unconfigured, returns nothing, or its search index is stale, it lists `Watch/` and matches the query against folder names, then reads each `report.md`. Needs `NEXTCLOUD_PASS`.

If both are unconfigured, the script says so. If a topic genuinely hasn't been watched, it reports no matches.

## Examples

```bash
# Find anything watched about beveling in Blender, snippet view
python3 "$RECALL" "bevel"

# Learn a whole tutorial back into context
python3 "$RECALL" "blender dice" --full

# Only the single best match
python3 "$RECALL" "subdivision surface" --limit 1 --full
```

## Troubleshooting

- **"No matching reports" but you know you watched it** → the Paperless full-text index may be stale (e.g. after a document was trashed and restored). The Nextcloud title fallback still works if you search by a word in the title. Rebuild the index with `docker exec paperless document_index reindex`.
- **"PAPERLESS_TOKEN not set / NEXTCLOUD_PASS not set"** → configure the same keys `/watch`'s library storage uses (`~/.config/watch/.env`). Recall reads from whichever targets are configured.

## Security & Permissions

- Reads only: it issues `GET`/`PROPFIND` against your configured Paperless (`PAPERLESS_URL`) and Nextcloud (`NEXTCLOUD_URL`) instances to fetch report text. It never writes, deletes, or uploads.
- Credentials come from `~/.config/watch/.env` (or `./.env`), the same file `/watch` uses. Nothing is logged or echoed.
- No video download, no frame extraction, no external API beyond your own Paperless/Nextcloud hosts.

**Bundled script:** `../watch/scripts/recall.py` (shares `library.py`'s config + Nextcloud/SSL helpers).
