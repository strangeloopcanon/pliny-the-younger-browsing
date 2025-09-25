#!/usr/bin/env python3
"""Inspect csv_sequences.jsonl or graph.json for quick stats."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict

from pliny_env.utils import domain_label


def load_sequences(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    sequences = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sequences.append(json.loads(line))
    return sequences


def summarize_sequences(sequences):
    total_steps = 0
    host_counter = Counter()
    domain_counter = Counter()
    for seq in sequences:
        total_steps += len(seq)
        if not seq:
            continue
        for step in seq:
            url = step.get("url", "")
            host = url.split("//")[-1].split("/")[0]
            host_counter[host] += 1
        domain = domain_label(seq[-1].get("url", ""), seq[-1].get("page_type", ""))
        domain_counter[domain] += 1
    return {
        "sequences": len(sequences),
        "avg_length": total_steps / max(len(sequences), 1),
        "top_hosts": host_counter.most_common(10),
        "domains": domain_counter.most_common(),
    }


def summarize_graph(path: Path) -> Dict:
    graph = json.loads(path.read_text())
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})
    outdeg = [len(v) for v in edges.values()]
    domain_counter = Counter(meta.get("domain") for meta in nodes.values() if meta.get("domain"))
    return {
        "nodes": len(nodes),
        "edges": sum(len(v) for v in edges.values()),
        "avg_out_degree": sum(outdeg) / max(len(outdeg), 1),
        "domain_counts": domain_counter.most_common(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="env_artifacts/csv_sequences.jsonl")
    parser.add_argument("--graph", default="env_artifacts/graph.json")
    args = parser.parse_args()

    seq_path = Path(args.sequences)
    graph_path = Path(args.graph)

    if seq_path.exists():
        sequences = load_sequences(seq_path)
        seq_summary = summarize_sequences(sequences)
        print("Sequences summary:")
        for key, value in seq_summary.items():
            print(f"  {key}: {value}")
    else:
        print(f"Sequences file not found: {seq_path}")

    if graph_path.exists():
        graph_summary = summarize_graph(graph_path)
        print("Graph summary:")
        for key, value in graph_summary.items():
            print(f"  {key}: {value}")
    else:
        print(f"Graph file not found: {graph_path}")


if __name__ == "__main__":
    main()
