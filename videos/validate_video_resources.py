#!/usr/bin/env python3
"""Validate and catalog the video and learning-resource subtree.

The script uses only Python's standard library. It validates the authored Part
files, reconciles the manual status registry, and optionally regenerates the
machine-readable catalog and review queues.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse

PART_GLOB = "Part_[0-9][0-9]_YoutubeVideos.md"
STATUS_FILE = "video_resource_status.csv"
GENERATED_FILES = (
    "video_catalog.csv",
    "video_catalog.json",
    "video_review_queue.csv",
    "unlinked_resource_candidates.csv",
    "duplicate_targets.csv",
    "video_summary.json",
)
EXPECTED_PARTS = set(range(1, 11))
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
HTTP_URL_RE = re.compile(r"https?://[^\s)\]>]+")
ANCHOR_RE = re.compile(r"<a\s+name=[\"']([^\"']+)[\"']\s*></a>", re.IGNORECASE)
TOC_LINK_RE = re.compile(r"\[([^\]]+)\]\(#([^)]+)\)")
PLACEHOLDER_RE = re.compile(
    r"(?i)(?:YOUR[_ -]?(?:URL|VIDEO)|PLACEHOLDER[_ -]?URL|INSERT[_ -]?(?:URL|LINK)|TODO\s*:\s*(?:URL|LINK))"
)
DURATION_RE = re.compile(
    r"(?i)(?:\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|hr\b|h\b)"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b|\bvariable\b|\bfull course(?: series)?\b"
    r"|\bmultiple (?:modules|tutorials|videos)\b|\beducational series\b)"
)
URLISH_RE = re.compile(r"^https?://", re.IGNORECASE)
YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

STATUS_FIELDS = [
    "file",
    "chapter",
    "entry",
    "status",
    "reviewed_on",
    "current_url",
    "previous_url",
    "issue",
    "candidate_title",
    "candidate_url",
    "candidate_confidence",
    "recommended_action",
    "evidence_url",
]

CATALOG_FIELDS = [
    "resource_id",
    "file",
    "part",
    "chapter",
    "chapter_topics",
    "entry_order_in_chapter",
    "entry_order_in_file",
    "title",
    "has_url",
    "url",
    "canonical_url",
    "target_type",
    "target_key",
    "youtube_id",
    "duration_text",
    "has_duration",
    "provider",
    "covers",
    "source_line",
    "status",
    "reviewed_on",
    "issue",
    "candidate_title",
    "candidate_url",
    "candidate_confidence",
    "recommended_action",
    "target_occurrences",
    "target_distinct_titles",
    "target_distinct_chapters",
]


@dataclass
class Finding:
    level: str
    code: str
    message: str


@dataclass
class Resource:
    file: str
    part: int
    chapter: str
    chapter_topics: str
    title: str
    source_line: int
    entry_order_in_chapter: int
    entry_order_in_file: int
    block_lines: list[str] = field(default_factory=list)
    url: str = ""
    canonical_url: str = ""
    target_type: str = "unlinked"
    target_key: str = ""
    youtube_id: str = ""
    duration_text: str = ""
    has_duration: bool = False
    provider: str = ""
    covers: str = ""
    target_occurrences: int = 0
    target_distinct_titles: int = 0
    target_distinct_chapters: int = 0
    status_record: dict[str, str] | None = None

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.file, self.chapter, self.title)

    @property
    def resource_id(self) -> str:
        raw = "\n".join(self.identity).encode("utf-8")
        return "VR-" + hashlib.sha256(raw).hexdigest()[:16].upper()

    @property
    def status(self) -> str:
        if self.status_record:
            return self.status_record["status"]
        return "assigned" if self.url else "unlinked_discovery_candidate"

    def catalog_row(self) -> dict[str, object]:
        record = self.status_record or {}
        return {
            "resource_id": self.resource_id,
            "file": self.file,
            "part": self.part,
            "chapter": self.chapter,
            "chapter_topics": self.chapter_topics,
            "entry_order_in_chapter": self.entry_order_in_chapter,
            "entry_order_in_file": self.entry_order_in_file,
            "title": self.title,
            "has_url": bool(self.url),
            "url": self.url,
            "canonical_url": self.canonical_url,
            "target_type": self.target_type,
            "target_key": self.target_key,
            "youtube_id": self.youtube_id,
            "duration_text": self.duration_text,
            "has_duration": self.has_duration,
            "provider": self.provider,
            "covers": self.covers,
            "source_line": self.source_line,
            "status": self.status,
            "reviewed_on": record.get("reviewed_on", ""),
            "issue": record.get("issue", ""),
            "candidate_title": record.get("candidate_title", ""),
            "candidate_url": record.get("candidate_url", ""),
            "candidate_confidence": record.get("candidate_confidence", ""),
            "recommended_action": record.get("recommended_action", ""),
            "target_occurrences": self.target_occurrences,
            "target_distinct_titles": self.target_distinct_titles,
            "target_distinct_chapters": self.target_distinct_chapters,
        }


def canonicalize_url(url: str) -> tuple[str, str, str, str]:
    """Return canonical URL, target key, target type, and YouTube ID."""
    url = url.strip().rstrip(".,;:")
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query = defaultdict(list)
    for key, value in query_pairs:
        query[key].append(value)

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch" and query.get("v"):
            video_id = query["v"][0]
            allowed = []
            for key in ("list", "t", "start", "end", "index"):
                for value in query.get(key, []):
                    allowed.append((key, value))
            canonical = f"https://www.youtube.com/watch?v={video_id}"
            if allowed:
                canonical += "&" + urlencode(allowed)
            return canonical, f"youtube:video:{video_id}", "youtube_video", video_id
        if parsed.path == "/playlist" and query.get("list"):
            playlist_id = query["list"][0]
            canonical = f"https://www.youtube.com/playlist?list={playlist_id}"
            return canonical, f"youtube:playlist:{playlist_id}", "youtube_playlist", ""
        if parsed.path.startswith("/shorts/"):
            parts = parsed.path.strip("/").split("/")
            video_id = parts[1] if len(parts) > 1 else ""
            canonical = f"https://www.youtube.com/watch?v={video_id}"
            return canonical, f"youtube:video:{video_id}", "youtube_video", video_id

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
        allowed = [(key, value) for key, value in query_pairs if key in {"list", "t", "start", "end", "index"}]
        canonical = f"https://www.youtube.com/watch?v={video_id}"
        if allowed:
            canonical += "&" + urlencode(allowed)
        return canonical, f"youtube:video:{video_id}", "youtube_video", video_id

    # Preserve web URLs as authored. Host/trailing-slash normalization is useful
    # for duplicate grouping but should not rewrite site-specific canonical URLs.
    canonical = url
    normalized_path = parsed.path or "/"
    if normalized_path != "/":
        normalized_path = normalized_path.rstrip("/")
    key = f"{host}{normalized_path}"
    if parsed.query:
        key += "?" + parsed.query
    return canonical, key, "web", ""


def duration_from_block(lines: list[str], url: str) -> tuple[str, bool]:
    candidates: list[str] = []
    for line in lines:
        stripped = line.strip()
        if url and url in stripped:
            close = stripped.rfind(")")
            if close >= 0:
                suffix = stripped[close + 1 :].strip(" -")
                if suffix:
                    candidates.append(suffix)
        if DURATION_RE.search(stripped):
            cleaned = stripped[2:].strip() if stripped.startswith("- ") else stripped
            if not URLISH_RE.match(cleaned) and not cleaned.lower().startswith("covers:"):
                candidates.append(cleaned)
    for candidate in candidates:
        if DURATION_RE.search(candidate):
            return candidate, True
    return "", False


def provider_and_covers(lines: list[str]) -> tuple[str, str]:
    providers: list[str] = []
    covers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        body = stripped[2:].strip()
        if body.lower().startswith("covers:"):
            covers.append(body.split(":", 1)[1].strip())
            continue
        if LINK_RE.search(body) or HTTP_URL_RE.search(body):
            continue
        if not body or DURATION_RE.fullmatch(body.strip("()~ ")):
            continue
        if DURATION_RE.search(body) and len(body.split()) <= 8:
            continue
        providers.append(body)
    return " | ".join(providers), " | ".join(covers)


def parse_part_file(path: Path, findings: list[Finding]) -> tuple[list[Resource], list[dict[str, object]]]:
    match = re.search(r"Part_(\d+)_", path.name)
    if not match:
        findings.append(Finding("error", "invalid_filename", f"Cannot determine part number from {path.name}."))
        return [], []
    part = int(match.group(1))
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    if not lines or not lines[0].startswith(f"# Part {part:02d}:"):
        findings.append(Finding("error", "invalid_h1", f"{path.name}: unexpected or missing H1."))
    elif "Video and Learning Resources" not in lines[0]:
        findings.append(Finding("warning", "legacy_h1", f"{path.name}: H1 does not use the broadened resource label."))

    if PLACEHOLDER_RE.search(text):
        findings.append(Finding("error", "placeholder_url", f"{path.name}: placeholder URL text remains."))

    explicit_anchors = set(ANCHOR_RE.findall(text))
    toc_links = TOC_LINK_RE.findall(text)
    for label, anchor in toc_links:
        if anchor not in explicit_anchors:
            findings.append(
                Finding("error", "missing_anchor", f"{path.name}: TOC link '{label}' points to missing anchor '#{anchor}'.")
            )

    for line_number, line in enumerate(lines, 1):
        for label, href in LINK_RE.findall(line):
            if URLISH_RE.match(label) and label != href:
                findings.append(
                    Finding(
                        "error",
                        "link_label_href_mismatch",
                        f"{path.name}:{line_number}: URL label and href differ: {label!r} vs {href!r}.",
                    )
                )
            if URLISH_RE.match(href):
                parsed = urlparse(href)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    findings.append(
                        Finding("error", "malformed_url", f"{path.name}:{line_number}: malformed URL {href!r}.")
                    )

    chapter = ""
    topics = ""
    chapter_entry_order = 0
    file_entry_order = 0
    resources: list[Resource] = []
    chapters: list[dict[str, object]] = []

    for index, line in enumerate(lines):
        if line.startswith("## ") and line.strip() != "## Table of Contents":
            chapter = line[3:].strip()
            topics = ""
            for probe in lines[index + 1 : min(index + 9, len(lines))]:
                if probe.startswith("**Topics:**"):
                    topics = probe.split("**Topics:**", 1)[1].strip()
                    break
            chapter_entry_order = 0
            chapters.append({"file": path.name, "part": part, "chapter": chapter, "source_line": index + 1, "topics": topics})
            continue

        if not line.startswith("### "):
            continue

        if not chapter:
            findings.append(
                Finding("error", "entry_without_chapter", f"{path.name}:{index + 1}: resource appears before any chapter.")
            )
        title = line[4:].strip()
        end = len(lines)
        for probe_index in range(index + 1, len(lines)):
            if lines[probe_index].startswith("### ") or lines[probe_index].startswith("## "):
                end = probe_index
                break
        block_lines = lines[index + 1 : end]
        links: list[str] = []
        for block_line in block_lines:
            for _, href in LINK_RE.findall(block_line):
                if URLISH_RE.match(href):
                    links.append(href.strip())
        if not links:
            for block_line in block_lines:
                links.extend(HTTP_URL_RE.findall(block_line))
        unique_links = list(dict.fromkeys(links))
        if len(unique_links) > 1:
            findings.append(
                Finding(
                    "error",
                    "multiple_resource_urls",
                    f"{path.name}:{index + 1}: '{title}' contains {len(unique_links)} external URLs; one canonical target is required.",
                )
            )
        url = unique_links[0] if unique_links else ""
        canonical_url = target_key = target_type = youtube_id = ""
        if url:
            canonical_url, target_key, target_type, youtube_id = canonicalize_url(url)
            if target_type == "youtube_video" and not YT_ID_RE.fullmatch(youtube_id):
                findings.append(
                    Finding(
                        "error",
                        "invalid_youtube_id",
                        f"{path.name}:{index + 1}: invalid YouTube video ID {youtube_id!r}.",
                    )
                )
            if target_type.startswith("youtube_") and url != canonical_url:
                findings.append(
                    Finding(
                        "warning",
                        "noncanonical_url",
                        f"{path.name}:{index + 1}: use canonical URL {canonical_url!r} instead of {url!r}.",
                    )
                )
        else:
            target_type = "unlinked"

        duration_text, has_duration = duration_from_block(block_lines, url)
        provider, covers = provider_and_covers(block_lines)
        chapter_entry_order += 1
        file_entry_order += 1
        resources.append(
            Resource(
                file=path.name,
                part=part,
                chapter=chapter,
                chapter_topics=topics,
                title=title,
                source_line=index + 1,
                entry_order_in_chapter=chapter_entry_order,
                entry_order_in_file=file_entry_order,
                block_lines=block_lines,
                url=url,
                canonical_url=canonical_url,
                target_type=target_type,
                target_key=target_key,
                youtube_id=youtube_id,
                duration_text=duration_text,
                has_duration=has_duration,
                provider=provider,
                covers=covers,
            )
        )

    heading_chapters = {chapter_row["chapter"] for chapter_row in chapters}
    if len(toc_links) != len(chapters):
        findings.append(
            Finding(
                "error",
                "toc_chapter_count_mismatch",
                f"{path.name}: {len(toc_links)} TOC chapter links but {len(chapters)} chapter headings.",
            )
        )
    toc_labels = {label for label, _ in toc_links}
    missing_toc = sorted(heading_chapters - toc_labels)
    if missing_toc:
        findings.append(
            Finding("error", "chapter_missing_from_toc", f"{path.name}: chapters missing from TOC: {', '.join(missing_toc)}")
        )

    return resources, chapters


def read_status_registry(root: Path, findings: list[Finding]) -> list[dict[str, str]]:
    path = root / STATUS_FILE
    if not path.exists():
        findings.append(Finding("error", "missing_status_registry", f"Missing {STATUS_FILE}."))
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != STATUS_FIELDS:
            findings.append(
                Finding(
                    "error",
                    "status_schema_mismatch",
                    f"{STATUS_FILE}: expected columns {STATUS_FIELDS}, found {reader.fieldnames}.",
                )
            )
        rows = list(reader)
    valid_statuses = {"resolved_official_replacement", "review_required", "verified"}
    for row_number, row in enumerate(rows, 2):
        if row.get("status") not in valid_statuses:
            findings.append(
                Finding(
                    "error",
                    "invalid_status",
                    f"{STATUS_FILE}:{row_number}: unsupported status {row.get('status')!r}.",
                )
            )
        for field_name in ("file", "chapter", "entry", "status", "reviewed_on", "current_url", "issue", "recommended_action"):
            if not row.get(field_name, "").strip():
                findings.append(
                    Finding(
                        "error",
                        "status_required_field",
                        f"{STATUS_FILE}:{row_number}: required field {field_name!r} is blank.",
                    )
                )
        for field_name in ("current_url", "previous_url", "candidate_url", "evidence_url"):
            value = row.get(field_name, "").strip()
            if value:
                parsed = urlparse(value)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    findings.append(
                        Finding(
                            "error",
                            "status_malformed_url",
                            f"{STATUS_FILE}:{row_number}: malformed {field_name} {value!r}.",
                        )
                    )
    return rows


def reconcile_status(resources: list[Resource], status_rows: list[dict[str, str]], findings: list[Finding]) -> None:
    by_identity: dict[tuple[str, str, str], list[Resource]] = defaultdict(list)
    for resource in resources:
        by_identity[resource.identity].append(resource)

    seen: set[tuple[str, str, str]] = set()
    for row_number, row in enumerate(status_rows, 2):
        identity = (row["file"], row["chapter"], row["entry"])
        if identity in seen:
            findings.append(
                Finding("error", "duplicate_status_identity", f"{STATUS_FILE}:{row_number}: duplicate identity {identity!r}.")
            )
            continue
        seen.add(identity)
        matches = by_identity.get(identity, [])
        if len(matches) != 1:
            findings.append(
                Finding(
                    "error",
                    "status_entry_not_found",
                    f"{STATUS_FILE}:{row_number}: expected one matching resource for {identity!r}, found {len(matches)}.",
                )
            )
            continue
        resource = matches[0]
        if resource.url != row["current_url"]:
            findings.append(
                Finding(
                    "error",
                    "status_url_mismatch",
                    f"{STATUS_FILE}:{row_number}: current_url {row['current_url']!r} does not match source URL {resource.url!r}.",
                )
            )
        resource.status_record = row


def enrich_duplicates(resources: list[Resource]) -> dict[str, list[Resource]]:
    groups: dict[str, list[Resource]] = defaultdict(list)
    for resource in resources:
        if resource.target_key:
            groups[resource.target_key].append(resource)
    for group in groups.values():
        occurrences = len(group)
        distinct_titles = len({resource.title for resource in group})
        distinct_chapters = len({resource.chapter for resource in group})
        for resource in group:
            resource.target_occurrences = occurrences
            resource.target_distinct_titles = distinct_titles
            resource.target_distinct_chapters = distinct_chapters
    return groups


def normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    """Write deterministic UTF-8 JSON with LF endings on every platform."""
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)


def write_outputs(root: Path, resources: list[Resource], chapters: list[dict[str, object]], groups: dict[str, list[Resource]]) -> dict[str, object]:
    catalog_rows = [resource.catalog_row() for resource in resources]
    write_csv(root / "video_catalog.csv", CATALOG_FIELDS, catalog_rows)
    write_json(root / "video_catalog.json", catalog_rows)

    linked_by_title: dict[str, list[Resource]] = defaultdict(list)
    for resource in resources:
        if resource.url:
            linked_by_title[normalized_title(resource.title)].append(resource)

    unlinked_fields = [
        "resource_id",
        "file",
        "part",
        "chapter",
        "title",
        "provider",
        "covers",
        "source_line",
        "internal_exact_match_count",
        "internal_exact_match_urls",
        "internal_exact_match_chapters",
    ]
    unlinked_rows = []
    for resource in resources:
        if resource.url:
            continue
        matches = linked_by_title.get(normalized_title(resource.title), [])
        unlinked_rows.append(
            {
                "resource_id": resource.resource_id,
                "file": resource.file,
                "part": resource.part,
                "chapter": resource.chapter,
                "title": resource.title,
                "provider": resource.provider,
                "covers": resource.covers,
                "source_line": resource.source_line,
                "internal_exact_match_count": len(matches),
                "internal_exact_match_urls": " | ".join(sorted({match.url for match in matches})),
                "internal_exact_match_chapters": " | ".join(sorted({match.chapter for match in matches})),
            }
        )
    write_csv(root / "unlinked_resource_candidates.csv", unlinked_fields, unlinked_rows)

    duplicate_fields = [
        "target_key",
        "target_type",
        "canonical_url",
        "occurrences",
        "distinct_titles",
        "distinct_chapters",
        "titles",
        "chapters",
        "files",
        "needs_scope_review",
    ]
    duplicate_rows = []
    for key, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(group) <= 1:
            continue
        titles = sorted({resource.title for resource in group})
        chapters_for_target = sorted({resource.chapter for resource in group})
        duplicate_rows.append(
            {
                "target_key": key,
                "target_type": group[0].target_type,
                "canonical_url": group[0].canonical_url,
                "occurrences": len(group),
                "distinct_titles": len(titles),
                "distinct_chapters": len(chapters_for_target),
                "titles": " | ".join(titles),
                "chapters": " | ".join(chapters_for_target),
                "files": " | ".join(sorted({resource.file for resource in group})),
                "needs_scope_review": len(titles) > 1,
            }
        )
    write_csv(root / "duplicate_targets.csv", duplicate_fields, duplicate_rows)

    review_fields = [
        "priority",
        "resource_id",
        "file",
        "chapter",
        "title",
        "current_url",
        "issue",
        "candidate_title",
        "candidate_url",
        "candidate_confidence",
        "recommended_action",
        "internal_exact_match_count",
        "internal_exact_match_urls",
    ]
    review_rows = []
    for resource in resources:
        matches = linked_by_title.get(normalized_title(resource.title), []) if not resource.url else []
        if resource.status == "review_required":
            record = resource.status_record or {}
            review_rows.append(
                {
                    "priority": "1_linked_target_unverified",
                    "resource_id": resource.resource_id,
                    "file": resource.file,
                    "chapter": resource.chapter,
                    "title": resource.title,
                    "current_url": resource.url,
                    "issue": record.get("issue", ""),
                    "candidate_title": record.get("candidate_title", ""),
                    "candidate_url": record.get("candidate_url", ""),
                    "candidate_confidence": record.get("candidate_confidence", ""),
                    "recommended_action": record.get("recommended_action", ""),
                    "internal_exact_match_count": "",
                    "internal_exact_match_urls": "",
                }
            )
        elif not resource.url:
            review_rows.append(
                {
                    "priority": "2_unlinked_discovery_candidate",
                    "resource_id": resource.resource_id,
                    "file": resource.file,
                    "chapter": resource.chapter,
                    "title": resource.title,
                    "current_url": "",
                    "issue": "No URL is assigned in the source file.",
                    "candidate_title": "",
                    "candidate_url": "",
                    "candidate_confidence": "",
                    "recommended_action": "Verify intent from chapter context before assigning a resource; do not auto-fill solely from title similarity.",
                    "internal_exact_match_count": len(matches),
                    "internal_exact_match_urls": " | ".join(sorted({match.url for match in matches})),
                }
            )
    write_csv(root / "video_review_queue.csv", review_fields, review_rows)

    by_part: dict[str, dict[str, int]] = {}
    for part in sorted({resource.part for resource in resources}):
        subset = [resource for resource in resources if resource.part == part]
        by_part[str(part)] = {
            "chapters": len({resource.chapter for resource in subset}),
            "entries": len(subset),
            "linked": sum(bool(resource.url) for resource in subset),
            "unlinked": sum(not resource.url for resource in subset),
            "review_required": sum(resource.status == "review_required" for resource in subset),
            "verified": sum(resource.status == "verified" for resource in subset),
        }

    summary = {
        "generated_by": "validate_video_resources.py",
        "source_files": len({resource.file for resource in resources}),
        "chapters": len(chapters),
        "entries": len(resources),
        "linked_entries": sum(bool(resource.url) for resource in resources),
        "unlinked_entries": sum(not resource.url for resource in resources),
        "entries_with_duration": sum(resource.has_duration for resource in resources),
        "entries_without_duration": sum(not resource.has_duration for resource in resources),
        "unique_targets": len(groups),
        "reused_targets": sum(len(group) > 1 for group in groups.values()),
        "targets_with_multiple_titles": sum(len({resource.title for resource in group}) > 1 for group in groups.values()),
        "resolved_official_replacements": sum(resource.status == "resolved_official_replacement" for resource in resources),
        "verified_entries": sum(resource.status == "verified" for resource in resources),
        "linked_targets_requiring_review": sum(resource.status == "review_required" for resource in resources),
        "unlinked_entries_with_internal_exact_title_match": sum(
            bool(linked_by_title.get(normalized_title(resource.title))) for resource in resources if not resource.url
        ),
        "review_queue_entries": len(review_rows),
        "by_part": by_part,
        "target_types": dict(Counter(resource.target_type for resource in resources if resource.url).most_common()),
        "hosts": dict(
            Counter(urlparse(resource.url).netloc.lower().removeprefix("www.") for resource in resources if resource.url).most_common()
        ),
    }
    write_json(root / "video_summary.json", summary)
    return summary


def check_generated_outputs(root: Path, findings: list[Finding]) -> bool:
    """Rebuild generated files in a temporary directory and compare bytes."""
    with tempfile.TemporaryDirectory(prefix="video-catalog-check-") as temp_dir:
        temp_root = Path(temp_dir)
        for source in sorted(root.glob(PART_GLOB)):
            shutil.copy2(source, temp_root / source.name)
        status_source = root / STATUS_FILE
        if status_source.exists():
            shutil.copy2(status_source, temp_root / STATUS_FILE)

        rebuilt_findings, _ = validate(temp_root, write=True)
        rebuild_errors = [item for item in rebuilt_findings if item.level == "error"]
        if rebuild_errors:
            findings.append(
                Finding(
                    "error",
                    "generated_rebuild_failed",
                    f"Could not rebuild generated catalog cleanly: {len(rebuild_errors)} validation error(s).",
                )
            )
            return False

        current = True
        for name in GENERATED_FILES:
            actual = root / name
            expected = temp_root / name
            if not actual.is_file():
                findings.append(Finding("error", "missing_generated_file", f"Missing generated file {name}."))
                current = False
            elif actual.read_bytes() != expected.read_bytes():
                findings.append(
                    Finding(
                        "error",
                        "stale_generated_file",
                        f"{name} is out of date; run this validator with `--write` (or run `make video-catalog`).",
                    )
                )
                current = False
        return current


def validate(root: Path, write: bool) -> tuple[list[Finding], dict[str, object]]:
    findings: list[Finding] = []
    part_files = sorted(root.glob(PART_GLOB))
    parts = {int(re.search(r"Part_(\d+)_", path.name).group(1)) for path in part_files}
    if parts != EXPECTED_PARTS:
        findings.append(
            Finding(
                "error",
                "part_set_mismatch",
                f"Expected parts {sorted(EXPECTED_PARTS)}, found {sorted(parts)} in {len(part_files)} files.",
            )
        )

    resources: list[Resource] = []
    chapters: list[dict[str, object]] = []
    for path in part_files:
        parsed_resources, parsed_chapters = parse_part_file(path, findings)
        resources.extend(parsed_resources)
        chapters.extend(parsed_chapters)

    identities = Counter(resource.identity for resource in resources)
    for identity, count in identities.items():
        if count > 1:
            findings.append(Finding("error", "duplicate_resource_identity", f"Resource identity {identity!r} appears {count} times."))

    status_rows = read_status_registry(root, findings)
    reconcile_status(resources, status_rows, findings)
    groups = enrich_duplicates(resources)

    unlinked_count = sum(not resource.url for resource in resources)
    missing_duration_count = sum(not resource.has_duration for resource in resources)
    review_count = sum(resource.status == "review_required" for resource in resources)
    multi_title_target_count = sum(len({resource.title for resource in group}) > 1 for group in groups.values())
    findings.extend(
        [
            Finding("warning", "unlinked_resources", f"{unlinked_count} resource entries have no assigned URL."),
            Finding("warning", "review_required", f"{review_count} linked resource entries remain in the explicit review queue."),
            Finding("warning", "missing_duration", f"{missing_duration_count} resource entries do not state a runtime or format length."),
            Finding("warning", "multi_title_reuse", f"{multi_title_target_count} reused targets appear under more than one displayed title."),
        ]
    )

    summary: dict[str, object]
    if write:
        summary = write_outputs(root, resources, chapters, groups)
    else:
        summary = {
            "source_files": len(part_files),
            "chapters": len(chapters),
            "entries": len(resources),
            "linked_entries": sum(bool(resource.url) for resource in resources),
            "unlinked_entries": unlinked_count,
            "linked_targets_requiring_review": review_count,
            "errors": sum(finding.level == "error" for finding in findings),
            "warnings": sum(finding.level == "warning" for finding in findings),
        }
    return findings, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Video subtree root.")
    parser.add_argument("--write", action="store_true", help="Regenerate catalog, duplicate, unlinked, review-queue, and summary files.")
    parser.add_argument("--check-generated", action="store_true", help="Fail when generated catalog files differ from a clean rebuild.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as a failing result.")
    parser.add_argument("--quiet", action="store_true", help="Print only the final summary line.")
    args = parser.parse_args()

    root = args.root.resolve()
    findings, summary = validate(root, args.write)
    if args.check_generated:
        summary["generated_outputs_current"] = check_generated_outputs(root, findings)
    errors = [finding for finding in findings if finding.level == "error"]
    warnings = [finding for finding in findings if finding.level == "warning"]

    if not args.quiet:
        for finding in findings:
            print(f"{finding.level.upper():7} {finding.code}: {finding.message}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    result = "PASS" if not errors and (not args.strict or not warnings) else "FAIL"
    print(f"VIDEO RESOURCE VALIDATION {result}: {len(errors)} error(s), {len(warnings)} warning(s).")
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
