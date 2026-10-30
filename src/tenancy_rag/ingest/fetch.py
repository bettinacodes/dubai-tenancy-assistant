"""Polite, cached acquisition of the official corpus.

Every fetch is written to data/raw/<source_id>/<source_id>__<YYYY-MM-DD>.<ext> and logged
to data/raw/manifest.jsonl (URL, SHA-256, timestamp, HTTP status, robots.txt verdict).
If a site refuses automated access we record that and stop: the fallback is a manual
download logged with acquisition="manual", never scraping around the block.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlsplit

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from tenancy_rag import config

SOURCES = config.DATA / "sources.json"
DELAY_SECONDS = 3.0  # between requests to the same host


def load_sources() -> list[dict]:
    return json.loads(SOURCES.read_text(encoding="utf-8"))


def robots_allows(url: str, client: httpx.Client) -> bool | None:
    """True/False from robots.txt; None if the site publishes none (no restriction)."""
    parts = urlsplit(url)
    resp = client.get(f"{parts.scheme}://{parts.netloc}/robots.txt")
    if resp.status_code != 200:
        return None
    rp = robotparser.RobotFileParser()
    rp.parse(resp.text.splitlines())
    return rp.can_fetch(config.USER_AGENT, url)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20), reraise=True)
def _get(client: httpx.Client, url: str) -> httpx.Response:
    return client.get(url)


def append_manifest(entry: dict) -> None:
    config.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def fetch_source(src: dict, client: httpx.Client) -> dict:
    url = src["url"]
    entry = {"source_id": src["id"], "url": url, "acquisition": "automated",
             "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    allowed = robots_allows(url, client)
    entry["robots_allows"] = allowed
    if allowed is False:
        entry.update(status=None, note="disallowed by robots.txt; download manually")
        return entry

    resp = _get(client, url)
    entry["status"] = resp.status_code
    if resp.status_code != 200:
        entry["note"] = "non-200 response; download manually if needed"
        return entry

    is_pdf = "pdf" in resp.headers.get("content-type", "") or url.lower().endswith(".pdf")
    ext = "pdf" if is_pdf else "html"
    out_dir = config.RAW / src["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{src['id']}__{entry['fetched_at'][:10]}.{ext}"
    path.write_bytes(resp.content)

    entry.update(path=str(path.relative_to(config.ROOT)).replace("\\", "/"),
                 sha256=hashlib.sha256(resp.content).hexdigest(),
                 bytes=len(resp.content),
                 content_type=resp.headers.get("content-type"),
                 encoding=resp.encoding)
    return entry


def fetch_all(only: set[str] | None = None) -> list[dict]:
    headers = {"User-Agent": config.USER_AGENT}
    results = []
    last_host = None
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        for src in load_sources():
            if only and src["id"] not in only:
                continue
            host = urlsplit(src["url"]).netloc
            if host == last_host:
                time.sleep(DELAY_SECONDS)
            last_host = host
            try:
                entry = fetch_source(src, client)
            except httpx.HTTPError as exc:
                entry = {"source_id": src["id"], "url": src["url"], "status": None,
                         "error": repr(exc), "note": "fetch failed; download manually"}
            append_manifest(entry)
            results.append(entry)
            print(f"{src['id']:32s} {entry.get('status')} {entry.get('bytes', '')} {entry.get('note', '')}")
    return results


if __name__ == "__main__":
    fetch_all()
