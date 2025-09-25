#!/usr/bin/env python3
"""CSV ingestion utilities for browsing logs."""

from __future__ import annotations

import csv
import io
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, TextIO

from .utils import canonicalize_url, clean_title, infer_page_type

logger = logging.getLogger(__name__)


@dataclass
class CsvEvent:
    org_id: str
    participant_id: str
    device_id: str
    url: str
    title: str
    transition: str
    event_time_iso: str
    event_timestamp: float
    visit_id: Optional[int]
    referring_visit_id: Optional[int]
    source_file: str
    page_type: str


@dataclass
class CsvIngestResult:
    nodes: Dict[str, Dict[str, Any]]
    edges: Dict[str, Counter]
    sequences: List[List[Dict[str, Any]]]
    sequence_path: Optional[str]
    meta: Dict[str, Any]


SECTION_SENTINELS = {"summary", "bookmarks", "cookies", "downloads", "history"}


def _parse_iso_timestamp(raw: str) -> Tuple[str, float]:
    if not raw:
        return "", 0.0
    val = raw.strip()
    if not val:
        return "", 0.0
    if val.endswith("Z"):
        val = val[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(val)
        return dt.isoformat(), dt.timestamp()
    except ValueError:
        return raw.strip(), 0.0


def _parse_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    val = str(raw).strip()
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _normalize_transition(raw: str) -> str:
    val = (raw or "").strip().lower()
    if not val:
        return "csv_navigate"
    return "csv_" + val.replace(" ", "_")


def _find_browsing_lines(text: str) -> List[str]:
    lines = text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "browsing":
            start_idx = idx + 1
            break
    if start_idx is None:
        return []
    # Skip empty lines until header
    while start_idx < len(lines) and not lines[start_idx].strip():
        start_idx += 1
    data_lines: List[str] = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in SECTION_SENTINELS:
            break
        data_lines.append(line)
    return data_lines


def _rows_from_lines(lines: List[str]) -> List[Dict[str, str]]:
    if not lines:
        return []
    csv_text = "\n".join(lines)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for row in reader:
        # Skip completely empty rows
        if not any(v and str(v).strip() for v in row.values()):
            continue
        rows.append(row)
    return rows


def _events_from_rows(rows: Iterable[Dict[str, str]], source: str) -> List[CsvEvent]:
    events: List[CsvEvent] = []
    for row in rows:
        url = canonicalize_url(row.get("url", ""))
        if not url:
            continue
        title = clean_title(row.get("title", ""))
        transition = _normalize_transition(row.get("transition", ""))
        iso, timestamp = _parse_iso_timestamp(row.get("eventtimeutc", ""))
        if not iso:
            iso, timestamp = _parse_iso_timestamp(row.get("eventtime", ""))
        visit_id = _parse_int(row.get("visitId"))
        referring_id = _parse_int(row.get("referringVisitId"))
        event = CsvEvent(
            org_id=row.get("OrgId", ""),
            participant_id=row.get("ParticipantId", ""),
            device_id=row.get("DeviceId", ""),
            url=url,
            title=title,
            transition=transition,
            event_time_iso=iso,
            event_timestamp=timestamp,
            visit_id=visit_id,
            referring_visit_id=referring_id,
            source_file=source,
            page_type=infer_page_type(url, title),
        )
        events.append(event)
    return events


def _add_edge(edges: Dict[str, Counter], src: str, etype: str, dst: str) -> None:
    if not src or not dst:
        return
    edges[src][(etype, dst)] += 1


def _aggregate_events(events: List[CsvEvent]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Counter], List[List[Dict[str, Any]]], Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Counter] = defaultdict(Counter)
    sequences: List[List[Dict[str, Any]]] = []
    participants = set()

    grouped: Dict[Tuple[str, str, str], List[CsvEvent]] = defaultdict(list)
    for event in events:
        key = (event.org_id, event.participant_id, event.device_id)
        grouped[key].append(event)
        participants.add(key)

    for key, seq_events in grouped.items():
        seq_events.sort(key=lambda e: (e.event_timestamp, e.event_time_iso, e.url))
        visit_map: Dict[int, str] = {}
        prev_url: Optional[str] = None
        seq_payload: List[Dict[str, Any]] = []
        for event in seq_events:
            url = event.url
            meta = nodes.setdefault(url, {"title": event.title, "page_type": event.page_type})
            if event.title and not meta.get("title"):
                meta["title"] = event.title
            if event.page_type and not meta.get("page_type"):
                meta["page_type"] = event.page_type

            if event.visit_id is not None:
                visit_map[event.visit_id] = url

            src_url: Optional[str] = None
            if event.referring_visit_id and event.referring_visit_id in visit_map:
                src_url = visit_map[event.referring_visit_id]
            elif prev_url and prev_url != url:
                src_url = prev_url

            if src_url and src_url != url:
                _add_edge(edges, src_url, event.transition, url)

            prev_url = url
            seq_payload.append(
                {
                    "url": url,
                    "title": event.title,
                    "page_type": event.page_type,
                    "transition": event.transition,
                    "source_file": event.source_file,
                    "event_time": event.event_time_iso,
                }
            )

        if len(seq_payload) >= 2:
            sequences.append(seq_payload)

    meta = {
        "participants": len(participants),
        "events": len(events),
    }
    return nodes, edges, sequences, meta


def ingest_csv_paths(
    paths: List[str],
    *,
    sequence_out_path: Optional[str] = None,
    append_sequences: bool = False,
    skip_paths: Optional[Set[str]] = None,
    collect_sequences: bool = False,
) -> CsvIngestResult:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Counter] = defaultdict(Counter)
    sequences: List[List[Dict[str, Any]]] = []
    meta = {
        "file_count": 0,
        "total_events": 0,
        "participants": 0,
        "files": [],
    }

    total_participants = 0

    writer: Optional[TextIO] = None
    written_sequences = 0
    if sequence_out_path:
        mode = "a" if append_sequences else "w"
        writer = open(sequence_out_path, mode, encoding="utf-8")

    skip_paths = skip_paths or set()

    try:
        for path in paths:
            if path in skip_paths:
                logger.info("Skipping CSV already processed: %s", path)
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except FileNotFoundError:
                logger.warning("CSV file not found: %s", path)
                continue

            data_lines = _find_browsing_lines(text)
            rows = _rows_from_lines(data_lines)
            events = _events_from_rows(rows, path)
            file_nodes, file_edges, file_sequences, file_meta = _aggregate_events(events)

            for url, meta_node in file_nodes.items():
                node = nodes.setdefault(url, {"title": meta_node.get("title", ""), "page_type": meta_node.get("page_type", "")})
                if meta_node.get("title") and not node.get("title"):
                    node["title"] = meta_node["title"]
                if meta_node.get("page_type") and not node.get("page_type"):
                    node["page_type"] = meta_node["page_type"]

            for src, counter in file_edges.items():
                edges[src].update(counter)

            if file_sequences:
                if writer is not None:
                    for seq in file_sequences:
                        json.dump(seq, writer)
                        writer.write("\n")
                        written_sequences += 1
                if collect_sequences:
                    sequences.extend(file_sequences)

            meta["file_count"] += 1
            meta["total_events"] += file_meta.get("events", 0)
            total_participants += file_meta.get("participants", 0)
            meta["files"].append({"path": path, "events": file_meta.get("events", 0)})

    finally:
        if writer is not None:
            writer.close()

    meta["participants"] = total_participants
    if sequence_out_path:
        meta["sequence_path"] = sequence_out_path
        meta["written_sequences"] = written_sequences

    return CsvIngestResult(
        nodes=nodes,
        edges=edges,
        sequences=sequences if collect_sequences else [],
        sequence_path=sequence_out_path,
        meta=meta,
    )
