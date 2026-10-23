#!/usr/bin/env python3
"""Route /watch artifacts into a browsable library: Immich (video + frames),
Nextcloud (report.md + any non-media), and Paperless-ngx (report.md, searchable).

- **video + representative frames -> Immich** (album per video: `POST
  /api/assets` multipart upload, dedup via a stable `deviceAssetId`, then
  `POST /api/albums` (or reuse an existing album by name) + `PUT
  /api/albums/{id}/assets`).
- **report.md (+ any non-media artifact) -> Nextcloud** (WebDAV folder per
  video: `MKCOL` then `PUT`, idempotent).
- **report.md (+ any non-media artifact) -> Paperless-ngx** (additive with
  Nextcloud: `POST /api/documents/post_document/`, tagged `watch`, so a
  no-vision agent can full-text-search across every watched video via the
  Paperless API. Relies on Paperless's own content-hash dedup for re-runs).

Pure stdlib (`urllib`, `json`, `base64`, `ssl`, `hashlib`, `mimetypes`).

Best-effort: `save_artifacts` and every uploader NEVER raise into the caller.
Any failure — missing config, a network error, a bad response — logs one
line to stderr and is recorded in the returned summary; the other target
still runs.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import uuid
from pathlib import Path
import sys as _sys

if str(Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config_file_path  # noqa: E402
from urllib.request import Request, urlopen

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

DEFAULT_IMMICH_BASE_URL = ""  # no built-in hosts: set them in ~/.config/watch/.env
DEFAULT_NEXTCLOUD_URL = ""
DEFAULT_NEXTCLOUD_USER = ""
DEFAULT_PAPERLESS_URL = ""
DEFAULT_FRAME_CAP = 20
PAPERLESS_TAG = "watch"

# Router: every artifact belongs to exactly one target. Media (video +
# frames) -> Immich; everything else (.md/.txt/.json/.vtt/.srt/...) -> Nextcloud.
IMMICH_MEDIA_EXT = {
    ".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi",
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
}

# select_representative_frames tiers (see below).
CUE_REASONS = {"cue", "transcript-cue"}
SCENE_REASONS = {"two-pass", "scene", "selected"}

CONFIG_KEYS = (
    "IMMICH_BASE_URL",
    "IMMICH_API_KEY",
    "NEXTCLOUD_URL",
    "NEXTCLOUD_USER",
    "NEXTCLOUD_PASS",
    "PAPERLESS_URL",
    "PAPERLESS_TOKEN",
    "WATCH_SAVE_FRAME_CAP",
)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def _from_dotenv(path: Path, name: str) -> str | None:
    """Copied from transcriptapi.load_api_key's dotenv parser: KEY=value,
    optional quotes, '#' comments, first match wins."""
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() != name:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            return value or None
    except OSError:
        return None
    return None


def _load_key(name: str) -> str | None:
    """env -> ~/.config/watch/.env -> ./.env, first non-empty wins."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    for candidate in (config_file_path(), Path.cwd() / ".env"):
        value = _from_dotenv(candidate, name)
        if value:
            return value
    return None


def load_config() -> dict:
    """Read the 6 library config keys (env -> ~/.config/watch/.env -> ./.env),
    applying defaults for base URLs, user, and frame cap."""
    cfg: dict = {key: _load_key(key) for key in CONFIG_KEYS}

    cfg["IMMICH_BASE_URL"] = cfg.get("IMMICH_BASE_URL") or DEFAULT_IMMICH_BASE_URL
    cfg["NEXTCLOUD_URL"] = cfg.get("NEXTCLOUD_URL") or DEFAULT_NEXTCLOUD_URL
    cfg["NEXTCLOUD_USER"] = cfg.get("NEXTCLOUD_USER") or DEFAULT_NEXTCLOUD_USER
    cfg["PAPERLESS_URL"] = cfg.get("PAPERLESS_URL") or DEFAULT_PAPERLESS_URL

    cap_raw = cfg.get("WATCH_SAVE_FRAME_CAP")
    try:
        cfg["WATCH_SAVE_FRAME_CAP"] = (
            int(cap_raw) if cap_raw not in (None, "") else DEFAULT_FRAME_CAP
        )
    except (TypeError, ValueError):
        cfg["WATCH_SAVE_FRAME_CAP"] = DEFAULT_FRAME_CAP

    return cfg


def save_exports(files: list[str], title: str, video_id: str, *, paperless_doc: str | None = None,
                 contact: str | None = None, cfg: dict | None = None, refs: dict | None = None) -> list[str]:
    """Route a run's exports to the same library entry as ``save_artifacts``.

    Every export goes to the Nextcloud folder ``Watch/<title> (<id>)``; ``paperless_doc``
    (the merged PDF, or report.md when no PDF was built) goes to Paperless; the contact
    sheet joins the Immich album. NEVER raises; ``refs`` gets ``paperless`` task ids.
    """
    summary: list[str] = []
    try:
        cfg = load_config() if cfg is None else (cfg or {})
        name = f"{sanitize(title)} ({video_id})"
        if files:
            if cfg.get("NEXTCLOUD_PASS"):
                r = upload_to_nextcloud(files, name, cfg)
                summary.append(f"Nextcloud: exports → Watch/{name} ({r.get('uploaded', 0)} files"
                               + (f", {r['error']})" if r.get("error") else ")"))
            else:
                summary.append("Nextcloud: exports skipped (no NEXTCLOUD_PASS)")
        if paperless_doc:
            if cfg.get("PAPERLESS_TOKEN"):
                r = upload_to_paperless([paperless_doc], name, cfg)
                if r.get("uploaded"):
                    _put_ref(refs, "paperless", {"task_ids": list(r.get("task_ids") or []), "url": r.get("url")})
                summary.append(f"Paperless: '{name}' ({Path(paperless_doc).name})"
                               + (f" ({r['error']})" if r.get("error") else ""))
            else:
                summary.append("Paperless: skipped (no PAPERLESS_TOKEN)")
        if contact:
            if cfg.get("IMMICH_API_KEY"):
                r = upload_to_immich([contact], name, cfg)
                summary.append(f"Immich: contact sheet → album '{name}'" + (f" ({r['error']})" if r.get("error") else ""))
            else:
                summary.append("Immich: contact sheet skipped (no IMMICH_API_KEY)")
    except Exception as exc:  # noqa: BLE001 - never fail the run
        summary.append(f"Library: exports not saved ({exc})")
    return summary


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

def is_immich_media(path: str) -> bool:
    """True when `path`'s extension is in Immich's media set (case-insensitive)."""
    if not path:
        return False
    return Path(path).suffix.lower() in IMMICH_MEDIA_EXT


def sanitize(title: str) -> str:
    """Make `title` safe as a filesystem/URL path segment; cap length ~80."""
    title = (title or "").strip()
    cleaned = re.sub(r'[\\/:*?"<>|]+', " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:80].rstrip()


# --------------------------------------------------------------------------
# Representative frame selection
# --------------------------------------------------------------------------

def _classify(frame: dict, ocr_text: dict) -> str:
    """ocr (has OCR text) -> cue (reason cue/transcript-cue) -> scene
    (reason two-pass/scene/selected) -> rest (everything else)."""
    path = frame.get("path") if isinstance(frame, dict) else None
    if path and (ocr_text.get(path) or "").strip():
        return "ocr"
    reason = frame.get("reason") if isinstance(frame, dict) else None
    if reason in CUE_REASONS:
        return "cue"
    if reason in SCENE_REASONS:
        return "scene"
    return "rest"


def _even_indices(n: int, k: int) -> list[int]:
    """k indices into range(n), evenly spaced, ascending, deduplicated."""
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    picked: list[int] = []
    for i in range(k):
        idx = round(i * (n - 1) / (k - 1))
        if idx not in picked:
            picked.append(idx)
    return picked


def select_representative_frames(frames: list[dict], ocr_text: dict, cap: int) -> list[dict]:
    """Priority OCR-significant -> cue -> scene -> even-spacing fill.

    `cap<=0` returns all frames. Ties within a tier keep timestamp order;
    the final list is timestamp-ordered.
    """
    ocr_text = ocr_text or {}
    frames_ts = sorted(frames or [], key=lambda f: f.get("timestamp_seconds", 0.0))

    try:
        cap = int(cap)
    except (TypeError, ValueError):
        cap = DEFAULT_FRAME_CAP

    if cap <= 0:
        return frames_ts

    tiers: dict[str, list[dict]] = {"ocr": [], "cue": [], "scene": [], "rest": []}
    for frame in frames_ts:
        tiers[_classify(frame, ocr_text)].append(frame)

    selected: list[dict] = []
    remaining = cap
    for tier_name in ("ocr", "cue", "scene"):
        if remaining <= 0:
            break
        tier = tiers[tier_name]
        if len(tier) > remaining:
            selected.extend(tier[:remaining])
            remaining = 0
        else:
            selected.extend(tier)
            remaining -= len(tier)

    if remaining > 0 and tiers["rest"]:
        rest = tiers["rest"]
        selected.extend(rest[i] for i in _even_indices(len(rest), remaining))

    selected.sort(key=lambda f: f.get("timestamp_seconds", 0.0))
    return selected


# --------------------------------------------------------------------------
# Multipart (adapted from whisper.py::_build_multipart — proven pattern)
# --------------------------------------------------------------------------

def _multipart(
    fields: dict,
    filename: str,
    filebytes: bytes,
    mimetype: str,
    file_field: str = "file",
) -> tuple[bytes, str]:
    """Assemble a multipart/form-data body. `file_field` is the form field
    name for the file part (Whisper APIs want "file"; Immich wants
    "assetData")."""
    boundary = f"----WatchBoundary{uuid.uuid4().hex}"
    eol = b"\r\n"
    buf = io.BytesIO()

    for name, value in fields.items():
        buf.write(f"--{boundary}".encode()); buf.write(eol)
        buf.write(f'Content-Disposition: form-data; name="{name}"'.encode()); buf.write(eol)
        buf.write(eol)
        buf.write(str(value).encode()); buf.write(eol)

    buf.write(f"--{boundary}".encode()); buf.write(eol)
    buf.write(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'.encode()
    )
    buf.write(eol)
    buf.write(f"Content-Type: {mimetype}".encode()); buf.write(eol)
    buf.write(eol)
    buf.write(filebytes)
    buf.write(eol)
    buf.write(f"--{boundary}--".encode()); buf.write(eol)

    return buf.getvalue(), boundary


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


def _iso8601(mtime: float) -> str:
    ms = int(round((mtime - int(mtime)) * 1000)) % 1000
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(mtime)) + f".{ms:03d}Z"


# --------------------------------------------------------------------------
# Immich uploader — standard Immich API flow
# --------------------------------------------------------------------------

def _device_asset_id(album_name: str, basename: str) -> str:
    """Stable per (album, filename) so re-running the same video does not
    duplicate assets (Immich dedups on deviceAssetId)."""
    digest = hashlib.sha256(f"{album_name}|{basename}".encode("utf-8")).hexdigest()[:16]
    return f"watch-{digest}-{basename}"


def _immich_headers(api_key: str, content_type: str | None = None) -> dict:
    headers = {
        "x-api-key": api_key,
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _upload_immich_asset(base_url: str, api_key: str, path: str, album_name: str) -> str | None:
    """POST one file to /api/assets. Returns the asset id, or None if the
    response carries none."""
    p = Path(path)
    data = p.read_bytes()
    mimetype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = time.time()
    iso = _iso8601(mtime)

    fields = {
        "deviceAssetId": _device_asset_id(album_name, p.name),
        "deviceId": "watch-skill",
        "fileCreatedAt": iso,
        "fileModifiedAt": iso,
    }
    body, boundary = _multipart(fields, p.name, data, mimetype, file_field="assetData")
    headers = _immich_headers(api_key, f"multipart/form-data; boundary={boundary}")
    request = Request(f"{base_url}/api/assets", data=body, headers=headers, method="POST")
    with urlopen(request, timeout=120, context=_ssl_context()) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
    return payload.get("id")


def _find_album(base_url: str, api_key: str, album_name: str) -> str | None:
    """Best-effort lookup of an existing album by exact name — avoids
    creating duplicate albums on re-run. Any failure returns None (caller
    falls back to creating a new album)."""
    try:
        request = Request(
            f"{base_url}/api/albums", headers=_immich_headers(api_key), method="GET"
        )
        with urlopen(request, timeout=30, context=_ssl_context()) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace") or "[]")
        if isinstance(data, list):
            for album in data:
                if isinstance(album, dict) and album.get("albumName") == album_name:
                    return album.get("id")
    except Exception:
        pass
    return None


def _create_album(base_url: str, api_key: str, album_name: str) -> str | None:
    body = json.dumps({"albumName": album_name}).encode("utf-8")
    request = Request(
        f"{base_url}/api/albums",
        data=body,
        headers=_immich_headers(api_key, "application/json"),
        method="POST",
    )
    with urlopen(request, timeout=30, context=_ssl_context()) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
    return data.get("id")


def _add_to_album(base_url: str, api_key: str, album_id: str, asset_ids: list[str]) -> None:
    body = json.dumps({"ids": asset_ids}).encode("utf-8")
    request = Request(
        f"{base_url}/api/albums/{album_id}/assets",
        data=body,
        headers=_immich_headers(api_key, "application/json"),
        method="PUT",
    )
    with urlopen(request, timeout=30, context=_ssl_context()) as response:
        response.read()


def upload_to_immich(files: list[str], album_name: str, cfg: dict) -> dict:
    """Upload files (video + frames) to an Immich album. NEVER raises."""
    result = {
        "album": album_name, "uploaded": 0, "url": None, "error": None,
        "album_id": None, "assets": {},
    }
    try:
        cfg = cfg or {}
        base_url = cfg.get("IMMICH_BASE_URL") or DEFAULT_IMMICH_BASE_URL
        api_key = cfg.get("IMMICH_API_KEY")
        if not api_key:
            result["error"] = "missing IMMICH_API_KEY"
            return result
        if not base_url:
            result["error"] = "missing IMMICH_BASE_URL"
            return result
        if not files:
            return result

        asset_ids: list[str] = []
        for path in files:
            try:
                asset_id = _upload_immich_asset(base_url, api_key, path, album_name)
                if asset_id:
                    asset_ids.append(asset_id)
                    result["assets"][str(path)] = asset_id
            except Exception as exc:
                print(f"[watch] library: Immich upload failed for {path}: {exc}", file=sys.stderr)

        result["uploaded"] = len(asset_ids)
        if not asset_ids:
            result["error"] = "no assets uploaded"
            return result

        album_id = None
        try:
            album_id = _find_album(base_url, api_key, album_name)
            if not album_id:
                album_id = _create_album(base_url, api_key, album_name)
        except Exception as exc:
            result["error"] = f"album create/lookup failed: {exc}"
            print(f"[watch] library: Immich {result['error']}", file=sys.stderr)

        result["album_id"] = album_id
        if album_id:
            try:
                _add_to_album(base_url, api_key, album_id, asset_ids)
                result["url"] = f"{base_url}/albums/{album_id}"
            except Exception as exc:
                result["error"] = f"album assign failed: {exc}"
                print(f"[watch] library: Immich {result['error']}", file=sys.stderr)

        return result
    except Exception as exc:
        result["error"] = str(exc)
        print(f"[watch] library: Immich upload failed: {exc}", file=sys.stderr)
        return result


# --------------------------------------------------------------------------
# Nextcloud uploader — WebDAV Basic auth, MKCOL then PUT
# --------------------------------------------------------------------------

def _nextcloud_auth_header(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _webdav_status(url: str, method: str, auth: str) -> int:
    """Issue a WebDAV request with no body, returning its HTTP status.
    HTTPError is treated as a status code, not an exception — MKCOL callers
    need to see 405/301 to treat an existing collection as OK."""
    request = Request(url, headers={"Authorization": auth}, method=method)
    try:
        with urlopen(request, timeout=60, context=_ssl_context()) as response:
            return getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        return exc.code


def _mkcol(url: str, auth: str) -> bool:
    """MKCOL a WebDAV collection; 405 (Method Not Allowed) or 301 on an
    already-existing collection is treated as success (idempotent)."""
    status = _webdav_status(url, "MKCOL", auth)
    return status in (200, 201, 405, 301)


def upload_to_nextcloud(files: list[str], folder: str, cfg: dict) -> dict:
    """Upload files to Nextcloud WebDAV under Watch/<folder>/. NEVER raises."""
    result = {"folder": folder, "uploaded": 0, "url": None, "error": None}
    try:
        cfg = cfg or {}
        base_url = cfg.get("NEXTCLOUD_URL") or DEFAULT_NEXTCLOUD_URL
        user = cfg.get("NEXTCLOUD_USER") or DEFAULT_NEXTCLOUD_USER
        password = cfg.get("NEXTCLOUD_PASS")
        if not password:
            result["error"] = "missing NEXTCLOUD_PASS"
            return result
        if not base_url or not user:
            result["error"] = "missing NEXTCLOUD_URL" if not base_url else "missing NEXTCLOUD_USER"
            return result
        if not files:
            return result

        auth = _nextcloud_auth_header(user, password)
        dav_root = f"{base_url}/remote.php/dav/files/{urllib.parse.quote(user)}"
        watch_dir = f"{dav_root}/Watch/"
        target_dir = f"{dav_root}/Watch/{urllib.parse.quote(folder)}/"

        try:
            _mkcol(watch_dir, auth)
        except Exception as exc:
            print(f"[watch] library: Nextcloud MKCOL Watch/ failed: {exc}", file=sys.stderr)
        try:
            _mkcol(target_dir, auth)
        except Exception as exc:
            print(f"[watch] library: Nextcloud MKCOL Watch/{folder}/ failed: {exc}", file=sys.stderr)

        uploaded = 0
        for path in files:
            try:
                p = Path(path)
                data = p.read_bytes()
                mimetype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                file_url = f"{target_dir}{urllib.parse.quote(p.name)}"
                request = Request(
                    file_url,
                    data=data,
                    headers={"Authorization": auth, "Content-Type": mimetype},
                    method="PUT",
                )
                with urlopen(request, timeout=120, context=_ssl_context()) as response:
                    response.read()
                uploaded += 1
            except Exception as exc:
                print(f"[watch] library: Nextcloud upload failed for {path}: {exc}", file=sys.stderr)

        result["uploaded"] = uploaded
        if uploaded == 0:
            result["error"] = "no files uploaded"
        return result
    except Exception as exc:
        result["error"] = str(exc)
        print(f"[watch] library: Nextcloud upload failed: {exc}", file=sys.stderr)
        return result


# --------------------------------------------------------------------------
# Paperless-ngx uploader — token auth, tag resolve, dedup, post_document
#
# report.md -> a searchable Paperless document (Tika ingests markdown as text),
# so a no-vision agent can later full-text-search across every watched video via
# the Paperless API. Additive with Nextcloud: Nextcloud keeps the browsable copy,
# Paperless provides the search index.
# --------------------------------------------------------------------------

def _paperless_headers(token: str, content_type: str | None = None) -> dict:
    headers = {
        "Authorization": f"Token {token}",
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _paperless_get(base_url: str, token: str, path: str) -> dict | list:
    request = Request(f"{base_url}{path}", headers=_paperless_headers(token), method="GET")
    with urlopen(request, timeout=30, context=_ssl_context()) as response:
        return json.loads(response.read().decode("utf-8", errors="replace") or "{}")


def _resolve_tag(base_url: str, token: str, name: str) -> int | None:
    """Return the id of tag `name`, creating it if absent. Best-effort:
    any failure returns None (the document is still uploaded, untagged)."""
    try:
        data = _paperless_get(
            base_url, token, f"/api/tags/?name__iexact={urllib.parse.quote(name)}"
        )
        results = data.get("results") if isinstance(data, dict) else None
        if results:
            return results[0].get("id")
    except Exception:
        pass
    try:
        body = json.dumps({"name": name}).encode("utf-8")
        request = Request(
            f"{base_url}/api/tags/",
            data=body,
            headers=_paperless_headers(token, "application/json"),
            method="POST",
        )
        with urlopen(request, timeout=30, context=_ssl_context()) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
        return payload.get("id")
    except Exception as exc:
        print(f"[watch] library: Paperless tag resolve failed: {exc}", file=sys.stderr)
        return None


def _post_document(
    base_url: str, token: str, path: str, title: str, tag_id: int | None
) -> tuple[bool, str | None]:
    """POST one file to /api/documents/post_document/. Returns (ok, task_id):
    ok is True on a 2xx submission (consumption is async, so this confirms the
    upload was accepted, not that the document finished processing); task_id is
    the consumption task id Paperless returns as a JSON string body, or None if
    the body is anything else."""
    p = Path(path)
    data = p.read_bytes()
    mimetype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"

    fields: dict = {"title": title}
    if tag_id is not None:
        fields["tags"] = tag_id

    body, boundary = _multipart(fields, p.name, data, mimetype, file_field="document")
    headers = _paperless_headers(token, f"multipart/form-data; boundary={boundary}")
    request = Request(
        f"{base_url}/api/documents/post_document/", data=body, headers=headers, method="POST"
    )
    with urlopen(request, timeout=120, context=_ssl_context()) as response:
        status = getattr(response, "status", 200)
        raw = response.read()
    ok = 200 <= status < 300
    task_id = None
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
        if isinstance(parsed, str) and parsed:
            task_id = parsed
    except Exception:
        pass
    return ok, task_id


def upload_to_paperless(files: list[str], title: str, cfg: dict) -> dict:
    """Upload non-media artifacts (report.md, ...) to Paperless-ngx as tagged,
    searchable documents. NEVER raises."""
    result = {"title": title, "uploaded": 0, "url": None, "error": None, "task_ids": []}
    try:
        cfg = cfg or {}
        base_url = (cfg.get("PAPERLESS_URL") or DEFAULT_PAPERLESS_URL).rstrip("/")
        token = cfg.get("PAPERLESS_TOKEN")
        if not token:
            result["error"] = "missing PAPERLESS_TOKEN"
            return result
        if not base_url:
            result["error"] = "missing PAPERLESS_URL"
            return result
        if not files:
            return result

        # No delete-then-reupload dedup: Paperless keeps deleted docs (and their
        # content hash) in a 30-day trash, so deleting a prior run's doc and
        # re-uploading identical bytes gets rejected as a duplicate — leaving
        # zero active docs. Paperless's own content-hash dedup already rejects
        # an identical re-run (the original survives), which is what we want.
        tag_id = _resolve_tag(base_url, token, PAPERLESS_TAG)

        uploaded = 0
        for path in files:
            try:
                ok, task_id = _post_document(base_url, token, path, title, tag_id)
                if ok:
                    uploaded += 1
                    if task_id:
                        result["task_ids"].append(task_id)
            except Exception as exc:
                print(f"[watch] library: Paperless upload failed for {path}: {exc}", file=sys.stderr)

        result["uploaded"] = uploaded
        if uploaded:
            result["url"] = f"{base_url}/documents?query={urllib.parse.quote(title)}"
        else:
            result["error"] = "no documents uploaded"
        return result
    except Exception as exc:
        result["error"] = str(exc)
        print(f"[watch] library: Paperless upload failed: {exc}", file=sys.stderr)
        return result


# --------------------------------------------------------------------------
# Router: content-type split + summary
# --------------------------------------------------------------------------

_REPORT_FRAME_LINE = re.compile(r"^- `([^`]+)`(.*)$", re.DOTALL)
_OCR_TAIL = " \u2014 OCR: "


def link_report_frames(report_path: str, frame_urls: dict[str, str]) -> int:
    """Rewrite the report in place: on each ``- `<path>` ...`` frame line whose
    path is in ``frame_urls``, insert `` \u2014 Immich: <url>`` before any
    `` \u2014 OCR: `` tail (else at end of line). Returns the number of lines
    changed. NEVER raises."""
    try:
        if not report_path or not frame_urls:
            return 0
        p = Path(report_path)
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        changed = 0
        for i, line in enumerate(lines):
            body = line.rstrip("\r\n")
            eol = line[len(body):]
            m = _REPORT_FRAME_LINE.match(body)
            if not m:
                continue
            url = frame_urls.get(m.group(1))
            if not url or " \u2014 Immich: " in body:
                continue
            link = f" \u2014 Immich: {url}"
            rest = m.group(2)
            head, sep, tail = rest.partition(_OCR_TAIL)
            lines[i] = f"- `{m.group(1)}`{head}{link}{sep}{tail}{eol}"
            changed += 1
        if changed:
            p.write_text("".join(lines), encoding="utf-8")
        return changed
    except Exception as exc:
        print(f"[watch] library: could not link report frames: {exc}", file=sys.stderr)
        return 0


def resolve_paperless_doc(
    task_id: str, cfg: dict, attempts: int = 3, delay: float = 1.0, sleep=time.sleep
) -> int | None:
    """Resolve a Paperless consumption task id to its document id via
    ``GET /api/tasks/?task_id=<id>``. Returns ``related_document`` (int) once
    the task reports SUCCESS, else None after ``attempts``. NEVER raises."""
    try:
        cfg = cfg or {}
        token = cfg.get("PAPERLESS_TOKEN")
        base_url = (cfg.get("PAPERLESS_URL") or DEFAULT_PAPERLESS_URL).rstrip("/")
        if not token or not task_id or not base_url:
            return None
        path = "/api/tasks/?task_id=" + urllib.parse.quote(str(task_id), safe="")
        for attempt in range(max(1, attempts)):
            try:
                data = _paperless_get(base_url, token, path)
                rows = data.get("results", []) if isinstance(data, dict) else data
                if rows and isinstance(rows[0], dict) and rows[0].get("status") == "SUCCESS":
                    doc = rows[0].get("related_document")
                    if doc not in (None, ""):
                        return int(doc)
            except Exception as exc:
                print(f"[watch] library: Paperless task lookup failed: {exc}", file=sys.stderr)
                return None
            if attempt < attempts - 1:
                sleep(delay)
        return None
    except Exception:
        return None


def _put_ref(refs: dict | None, key: str, value) -> None:
    """Best-effort: record one target's refs. Never raises, never affects the
    summary — a bad refs argument must not turn a good save into a failure."""
    try:
        if isinstance(refs, dict):
            refs[key] = value
    except Exception as exc:
        print(f"[watch] library: could not record refs[{key}]: {exc}", file=sys.stderr)


def save_artifacts(
    video_path: str | None,
    frames: list[dict],
    report_path: str | None,
    ocr_text: dict,
    title: str,
    video_id: str,
    cfg: dict | None = None,
    refs: dict | None = None,
    paperless: bool = True,
) -> list[str]:
    """Route this run's artifacts to Immich (video + representative frames)
    and Nextcloud (report.md + any non-media). NEVER raises — any failure is
    logged to stderr and reflected in the returned summary.

    Returns human-readable summary lines for the report footer.

    When ``refs`` is a dict it is filled in place with machine-readable
    pointers (``name``, ``immich`` album/asset ids, ``nextcloud`` folder,
    ``paperless`` task ids, each with its ``url``) for every target that
    actually uploaded something; skipped/failed targets are absent. The
    return value is unaffected.

    ``paperless=False`` defers the Paperless upload to ``save_exports`` (the run
    will produce a PDF that replaces report.md there).
    """
    try:
        if cfg is None:
            cfg = load_config()
        cfg = cfg or {}

        name = f"{sanitize(title)} ({video_id})"
        cap = cfg.get("WATCH_SAVE_FRAME_CAP", DEFAULT_FRAME_CAP)

        try:
            selected_frames = select_representative_frames(frames or [], ocr_text or {}, cap)
        except Exception as exc:
            print(f"[watch] library: frame selection failed, using all frames: {exc}", file=sys.stderr)
            selected_frames = list(frames or [])

        immich_files: list[str] = []
        if video_path and is_immich_media(video_path):
            immich_files.append(video_path)
        for frame in selected_frames:
            path = frame.get("path") if isinstance(frame, dict) else None
            if path and is_immich_media(path):
                immich_files.append(path)

        nextcloud_files: list[str] = []
        if report_path:
            if is_immich_media(report_path):
                immich_files.append(report_path)
            else:
                nextcloud_files.append(report_path)

        summary: list[str] = []

        if immich_files:
            if cfg.get("IMMICH_API_KEY"):
                try:
                    result = upload_to_immich(immich_files, name, cfg)
                except Exception as exc:
                    result = {"error": str(exc), "uploaded": 0}
                    print(f"[watch] library: Immich uploader raised: {exc}", file=sys.stderr)
                uploaded = result.get("uploaded", 0)
                error = result.get("error")
                path_assets = result.get("assets") or {}
                if uploaded:
                    frames_by_t_ms: dict[str, str] = {}
                    frame_urls: dict[str, str] = {}
                    immich_base = (cfg.get("IMMICH_BASE_URL") or DEFAULT_IMMICH_BASE_URL).rstrip("/")
                    for frame in frames or []:
                        try:
                            fpath = str(frame.get("path"))
                            asset_id = path_assets.get(fpath)
                            if not asset_id:
                                continue
                            ts = frame.get("timestamp_seconds")
                            if ts is not None:
                                frames_by_t_ms[str(int(round(float(ts) * 1000)))] = asset_id
                            frame_urls[fpath] = f"{immich_base}/photos/{asset_id}"
                        except Exception:
                            continue
                    _put_ref(refs, "immich", {
                        "album_id": result.get("album_id"),
                        "url": result.get("url"),
                        "assets": {Path(k).name: v for k, v in path_assets.items()},
                        "frames": frames_by_t_ms,
                    })
                    if report_path and frame_urls:
                        link_report_frames(report_path, frame_urls)
                if error and not uploaded:
                    summary.append(f"Immich: error ({error})")
                elif error:
                    summary.append(f"Immich: album '{name}' ({uploaded} assets, {error})")
                else:
                    summary.append(f"Immich: album '{name}' ({uploaded} assets)")
            else:
                summary.append("Immich: skipped (no IMMICH_API_KEY)")

        if nextcloud_files:
            if cfg.get("NEXTCLOUD_PASS"):
                try:
                    result = upload_to_nextcloud(nextcloud_files, name, cfg)
                except Exception as exc:
                    result = {"error": str(exc), "uploaded": 0}
                    print(f"[watch] library: Nextcloud uploader raised: {exc}", file=sys.stderr)
                uploaded = result.get("uploaded", 0)
                error = result.get("error")
                if uploaded:
                    _put_ref(refs, "nextcloud", {
                        "folder": result.get("folder") or name,
                        "url": result.get("url"),
                    })
                if error and not uploaded:
                    summary.append(f"Nextcloud: error ({error})")
                elif error:
                    summary.append(f"Nextcloud: Watch/{name} ({uploaded} files, {error})")
                else:
                    summary.append(f"Nextcloud: Watch/{name} ({uploaded} files)")
            else:
                summary.append("Nextcloud: skipped (no NEXTCLOUD_PASS)")

        # Paperless is additive with Nextcloud: same non-media artifacts, but
        # indexed for full-text search. Only report.md-class files, never media.
        if nextcloud_files and paperless:
            if cfg.get("PAPERLESS_TOKEN"):
                try:
                    result = upload_to_paperless(nextcloud_files, name, cfg)
                except Exception as exc:
                    result = {"error": str(exc), "uploaded": 0}
                    print(f"[watch] library: Paperless uploader raised: {exc}", file=sys.stderr)
                uploaded = result.get("uploaded", 0)
                error = result.get("error")
                if uploaded:
                    _put_ref(refs, "paperless", {
                        "task_ids": list(result.get("task_ids") or []),
                        "url": result.get("url"),
                    })
                if error and not uploaded:
                    summary.append(f"Paperless: error ({error})")
                elif error:
                    summary.append(f"Paperless: '{name}' ({uploaded} docs, {error})")
                else:
                    summary.append(f"Paperless: '{name}' ({uploaded} docs)")
            else:
                summary.append("Paperless: skipped (no PAPERLESS_TOKEN)")

        if refs:
            _put_ref(refs, "name", name)

        if not summary:
            summary.append("Library: nothing to save")

        return summary
    except Exception as exc:
        try:
            print(f"[watch] library save_artifacts failed: {exc}", file=sys.stderr)
        except Exception:
            pass
        return [f"Library: save failed ({exc})"]
