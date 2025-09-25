#!/usr/bin/env python3
"""Graph builder for browsing trajectories and CSV logs."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .csv_ingest import CsvIngestResult, ingest_csv_paths
from .utils import canonicalize_url

logger = logging.getLogger(__name__)


@dataclass
class GraphBuildConfig:
    trajectories_path: str = "corrected_trajectories.json"
    raw_csv_paths: Optional[List[str]] = None
    output_dir: str = "env_artifacts"
    graph_file: str = "graph.json"
    min_edge_count: int = 1
    base_graph_path: Optional[str] = None
    sequence_out_path: Optional[str] = None
    append_sequences: bool = False
    skip_csv_paths: Optional[Set[str]] = None


def _add_edge(edges: Dict[str, Counter], src: str, etype: str, dst: str) -> None:
    if not src or not dst or not etype:
        return
    edges[src][(etype, dst)] += 1


def _load_trajectories(path: str) -> List[List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("trajectories", [])


def _ingest_json(trajs: List[List[Dict[str, Any]]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Counter]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Counter] = defaultdict(Counter)
    for traj in trajs:
        for step in traj:
            s = step.get("state", {})
            a = step.get("action", {})
            src = canonicalize_url(s.get("url", ""))
            if src and src not in nodes:
                nodes[src] = {
                    "title": s.get("title", ""),
                    "page_type": s.get("page_type", "") or "generic_web_page",
                }
            etype = a.get("type", "") or "navigate"
            dst = canonicalize_url(a.get("target_url", ""))
            _add_edge(edges, src, etype, dst)
    return nodes, edges


def merge_graphs(
    base_nodes: Dict[str, Dict[str, Any]],
    base_edges: Dict[str, Counter],
    extra_nodes: Dict[str, Dict[str, Any]],
    extra_edges: Dict[str, Counter],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Counter]]:
    nodes = dict(base_nodes)
    for url, meta in extra_nodes.items():
        if url not in nodes:
            nodes[url] = meta
        else:
            if not nodes[url].get("title") and meta.get("title"):
                nodes[url]["title"] = meta["title"]
            if not nodes[url].get("page_type") and meta.get("page_type"):
                nodes[url]["page_type"] = meta["page_type"]

    edges: Dict[str, Counter] = defaultdict(Counter)
    for src, counter in base_edges.items():
        edges[src].update(counter)
    for src, counter in extra_edges.items():
        edges[src].update(counter)
    return nodes, edges


def build_graph(cfg: GraphBuildConfig) -> Tuple[Dict[str, Any], Optional[CsvIngestResult]]:
    os.makedirs(cfg.output_dir, exist_ok=True)

    nodes_json, edges_json = _ingest_json(_load_trajectories(cfg.trajectories_path))

    base_nodes: Dict[str, Dict[str, Any]] = {}
    base_edges: Dict[str, Counter] = defaultdict(Counter)
    base_meta: Dict[str, Any] = {}
    if cfg.base_graph_path and os.path.exists(cfg.base_graph_path):
        with open(cfg.base_graph_path, "r", encoding="utf-8") as f:
            base_graph = json.load(f)
        base_nodes = base_graph.get("nodes", {})
        for src, items in base_graph.get("edges", {}).items():
            for item in items:
                base_edges[src][(item.get("type", "navigate"), item.get("target", ""))] += int(item.get("count", 1))
        base_meta = base_graph.get("meta", {})

    csv_result: Optional[CsvIngestResult] = None
    nodes_csv: Dict[str, Dict[str, Any]] = {}
    edges_csv: Dict[str, Counter] = defaultdict(Counter)
    if cfg.raw_csv_paths:
        csv_result = ingest_csv_paths(
            cfg.raw_csv_paths,
            sequence_out_path=cfg.sequence_out_path,
            append_sequences=cfg.append_sequences,
            skip_paths=cfg.skip_csv_paths,
            collect_sequences=cfg.sequence_out_path is None,
        )
        nodes_csv = csv_result.nodes
        edges_csv = csv_result.edges

    intermediate_nodes, intermediate_edges = merge_graphs(nodes_json, edges_json, nodes_csv, edges_csv)
    nodes, edges = merge_graphs(base_nodes, base_edges, intermediate_nodes, intermediate_edges)

    compact_edges: Dict[str, List[Dict[str, Any]]] = {}
    for src, counter in edges.items():
        items = [
            {"type": etype, "target": dst, "count": cnt}
            for (etype, dst), cnt in counter.items()
            if cnt >= cfg.min_edge_count and dst
        ]
        items.sort(key=lambda x: (-x["count"], x["type"], x["target"]))
        if items:
            compact_edges[src] = items

    meta: Dict[str, Any] = {
        "source_json": cfg.trajectories_path,
        "source_csv": cfg.raw_csv_paths,
        "num_nodes": len(nodes),
        "num_edges": sum(len(v) for v in compact_edges.values()),
    }
    if base_meta:
        base_csv_meta = base_meta.get("csv_meta")
        if base_csv_meta:
            meta["csv_meta"] = {
                key: list(value) if isinstance(value, list) else value
                for key, value in base_csv_meta.items()
            }
        if "source_csv" not in meta or meta["source_csv"] is None:
            meta["source_csv"] = base_meta.get("source_csv")
        if base_meta.get("csv_sequence_file"):
            meta["csv_sequence_file"] = base_meta["csv_sequence_file"]
    meta.setdefault("csv_meta", {})

    graph = {
        "nodes": nodes,
        "edges": compact_edges,
        "meta": meta,
    }

    if csv_result and (csv_result.sequences or csv_result.sequence_path):
        sequence_path = csv_result.sequence_path or os.path.join(cfg.output_dir, "csv_sequences.jsonl")
        meta["csv_sequence_file"] = sequence_path
        csv_meta = meta.setdefault("csv_meta", {})
        existing_files = list(csv_meta.get("files", []))
        csv_meta_files = existing_files
        csv_meta["file_count"] = csv_meta.get("file_count", 0)
        csv_meta["total_events"] = csv_meta.get("total_events", 0)
        csv_meta["participants"] = csv_meta.get("participants", 0)

        file_entries = csv_result.meta.get("files", [])
        if file_entries:
            csv_meta_files.extend(file_entries)
            csv_meta["file_count"] += len(file_entries)
        csv_meta["total_events"] += csv_result.meta.get("total_events", 0)
        csv_meta["participants"] += csv_result.meta.get("participants", 0)
        if csv_result.meta.get("written_sequences") is not None:
            csv_meta["written_sequences"] = csv_meta.get("written_sequences", 0) + csv_result.meta.get("written_sequences", 0)
        csv_meta["files"] = csv_meta_files

    graph["meta"] = meta

    out_path = os.path.join(cfg.output_dir, cfg.graph_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f)
    logger.info("Wrote graph: %s", out_path)

    return graph, csv_result


def load_graph(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
