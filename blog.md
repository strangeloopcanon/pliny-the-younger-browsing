# Pliny the Younger: A Reproducible Web‑Browsing Environment

Most “web agents” are hard to compare fairly: every demo runs on a different set of pages, with different browser states, and non‑deterministic site behavior. Pliny the Younger approaches evaluation from the other end: build a deterministic, data‑driven environment that looks like the web and is identical for every run. Models then act inside this fixed world, so differences in scores come from the agent—not the environment.

This post focuses on the environment itself: how the graph is built, why it makes for a solid evaluation harness, and where we can take it next to get closer to a live web.

## A Graph, Not A Browser

Instead of automating a real browser, Pliny exposes a directed graph constructed from recorded browsing. Each node is a URL with light metadata (title, page type). Each outgoing edge represents a user‑observed transition from that URL to another target, labeled by an action type (e.g., navigate, csv_link, back). Agents receive a numbered action menu derived from these edges plus READ and STOP, and step through episodes until they reach a goal URL or exhaust a step budget.

This design trades surface realism for control and repeatability:
- Deterministic: the graph never changes during evaluation.
- Comparable: every agent sees the exact same states, actions, and goals.
- Inspectable: artifacts are plain JSON; you can audit nodes, edges, and tasks.
- Efficient: no DOM, JS, network, or site flakiness in the loop.

## How The Graph Gets Built

The environment is built from two data sources, then merged and compacted into a single `graph.json` under `env_artifacts/`.

1) Corrected trajectories (JSON)
- Source: `corrected_trajectories.json`.
- For each step, we canonicalize the current URL, record its title/page_type, and add an edge from the current URL to the action’s `target_url` (defaulting the action type to `navigate` if missing).

2) Raw browsing logs (CSV)
- Source: one or more CSV exports. We scan the “Browsing” section, parse columns (URL, title, transition, event time, visit IDs), normalize transition labels (`csv_*`), and group lines by participant/session.
- Within each session we sort by timestamp, resolve `referring_visit_id` when present (fall back to the previous URL), and add edges for cross‑URL transitions. We also emit per‑session sequences to `csv_sequences.jsonl` so tasks can later be sampled from these natural paths.

Merge and compact
- JSON and CSV graphs are merged with simple precedence rules for missing titles/page types. Edge counts are aggregated, then compacted to per‑source lists of `{type, target, count}` (optionally filtering by minimum count). Metadata tracks sources and CSV ingestion stats.
- A resumable mode skips already‑processed CSVs and appends new sequences safely.

Artifacts produced by `scripts/build_env_graph.py`
- `graph.json`: nodes, typed edges with counts, and metadata.
- `tasks_train.json` / `tasks_test.json`: start→goal tasks (with optional reference paths).
- `csv_sequences.jsonl`: per‑participant sequences streamed from CSVs.
- `env_config.json`: a small pointer file the environment loader reads.

### Artifact Flow

```
corrected_trajectories.json       CSV logs (one or many)
            \                         /
             \   ingest + merge + compact   (resume-safe)
              \         build_env_graph.py         
               \               |                  
                +--> env_artifacts/graph.json     (nodes + typed edges + meta)
                +--> env_artifacts/csv_sequences.jsonl (per-session sequences)
                +--> env_artifacts/tasks_train.json    (tasks from JSON + CSV)
                +--> env_artifacts/tasks_test.json
                +--> env_artifacts/env_config.json     (pointers + defaults)

env_artifacts/env_config.json --> PlinyBrowseEnv --> demo/eval/server
                                                \-> scripts/demo_env.py
                                                \-> scripts/evaluate_policies.py
                                                \-> scripts/env_server.py
```

Tips
- Inspect a built graph quickly with `scripts/inspect_sequences.py`.
- Use `--resume` on subsequent builds to skip previously processed CSVs and append sequences safely.

Quick build example
```
python scripts/build_env_graph.py \
  --traj corrected_trajectories.json \
  --csv data/my_log.csv \
  --csv-glob "data/*.csv" \
  --resume \
  --out env_artifacts
```

## Tasks, Observations, And Episodes

Tasks
- A task specifies `start_url`, `goal_url`, and an optional `reference_path` (a recorded route). Task difficulty is derived from path length. You can filter tasks by domain/host and hop limits to control diversity and difficulty.

Observations
- At each step, the agent sees the current page metadata, a short history window, the goal text, and a numbered list of available actions (top‑K outgoing edges). Optional extras include a light “REFLECTION” prompt and a list of alternative candidates. Observations can be fully templated with Jinja for consistent prompt layout across engines.

Episodes
- Transitions follow the chosen edge; READ keeps you in place; STOP ends the episode. Episodes terminate on reaching the goal or hitting `max_steps`. Rewards are shaped consistently (step penalties, terminal success, optional bonuses for following the reference path), enabling like‑for‑like comparisons across agents.

## Why This Makes A Good Evaluation Harness

- Reproducible world: A fixed graph and task set remove network and site nondeterminism.
- Shared tasks with ground truth: Reference paths provide an anchor for path‑quality metrics (e.g., path‑length ratio) in addition to success/length/reward.
- Tunable difficulty: Filter by hops/domains and cap action menus (`top_k`) to create controlled challenge regimes.
- Transparent artifacts: Everything is JSON; you can diff, subset, and audit inputs/outputs.
- Portable: Evaluations can run on laptops or CI without heavy browser stacks.

This is intentionally not a pixel‑perfect web emulator. It captures navigation structure—not rendering, auth, or dynamic content. That limitation is the point: for many agentic tasks, we first want consistent navigation benchmarks before we test real‑world robustness.

## Limitations (Today)

- No live browser: there’s no DOM/JS execution, layout, or cookies/auth state.
- Coarse action semantics: edges are typed but treated uniformly by the environment.
- Data‑bound coverage: you can only traverse transitions seen in the logs/trajectories.

These are acceptable trade‑offs for a primary goal of reproducible comparison.

## A Path To “Live” While Staying Reproducible

Here’s a practical roadmap to evolve toward a live web without losing comparability:

1) Snapshot‑driven eval
- Host per‑task HTML/asset snapshots (WARC or self‑contained bundles). Drive headless browsers (Playwright) against these snapshots with network replay. Deterministic content; DOM‑aware actions; identical for every run.

2) Hybrid graph+browser mode
- Keep the graph as the action space oracle. Execute chosen actions in a headless browser restricted to URLs inside the graph. When the DOM diverges, fall back to the graph edge target to keep evaluation deterministic.

3) Record/replay harness for live sites
- For selected sites, proxy and cache responses; lock user accounts/fixtures; time‑box eval windows. Record all requests/DOM states to enable replay across agents and reruns.

4) Richer actions
- Expand edges to cover form fills and structured interactions; store field schemas and selectors with targets. Continue emitting text‑only observations for cross‑engine parity.

### Deterministic “Live” Info (Use/React)

If the goal is to force models to read page content and react to it (e.g., “pick the cheapest plan”, “click the article mentioning X”), we can add a deterministic information layer without sacrificing reproducibility:

- Per‑node content payloads
  - Extend each node in `graph.json` with a small, structured `content` object (e.g., `{price, rating, headline, code}`), generated deterministically from `(task_id, url, seed)` or loaded from a versioned dataset.
  - Render this `content` in observations via the existing Jinja template support, so every policy sees the same text blocks.

- Constraint‑driven tasks
  - Encode task objectives that reference these fields (“reach the product page with the lowest `price` among candidates”, “choose the post whose `headline` contains 'release'”).
  - Keep a reference path consistent with the constraint so success remains checkable.

- Local “info API” for live‑ish data
  - Optionally, back `content` with a tiny local HTTP service keyed by `(task_id, url)` that returns deterministic JSON/text. The env fetches it at step time and injects into the observation. Still offline/replayable; no external network.

- Validation hooks
  - Use the custom reward plugin (see `reward.reward_module` / `reward.reward_class`) to verify that the chosen action satisfies the content‑based constraint (e.g., clicked the min‑price option). This keeps scoring objective and consistent.

Trade‑off: this reduces realism versus scraping real, changing pages, but it usefully tests grounding and selection skills under controlled noise. Later, the same pattern can swap in snapshot‑extracted DOM snippets instead of synthetic fields for higher fidelity.

Each step preserves the core property: different agents are judged on the same tasks, states, and semantics.

## Try The Environment

- Build artifacts: `python scripts/build_env_graph.py ...`
- Run a demo episode: `python scripts/demo_env.py --cfg env_artifacts/env_config.json`
- Expose over HTTP for external loops: `python scripts/env_server.py --env-config env_artifacts/env_config.json`

That’s enough to evaluate a variety of agents against a stable, inspectable browsing world. Training loops and model backends plug in on top—but the evaluation bedrock is the graph.
