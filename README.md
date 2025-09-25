# Pliny RL Environment & GSPO Training

This repository provides a small, reproducible RL stack for web‑browsing style tasks. It is intended as a canonical evaluation sandbox for foundation-model labs: drop in your engine, run the shared environment, and report comparable rewards. The stack contains:

1. A graph-based browsing environment constructed from trajectory logs (JSON + optional CSV dumps).
2. A GSPO (Group Sequence Policy Optimisation) trainer with interchangeable policies:
   - `hash`: lightweight hashed softmax (baseline, trainable).
   - `hf`: HuggingFace Transformers models (trainable via PyTorch).
   - `mlx`: MLX models driven through [`mlx-genkit`](https://github.com/strangeloopcanon/mlx-genkit) (inference/offline scoring on Apple Silicon).

The codebase is modular: environments live in `pliny_env/`, reusable RL plumbing in `rl/`, and entry points in `scripts/` and `train_gspo.py`.

---

## 1. Setup

Requirements (tested on macOS ARM64 / Python 3.11):

```bash
pip install -U \
  torch transformers accelerate datasets \
  mlx mlx-metal mlx-lm mlx-genkit \
  verifiers fastapi uvicorn
```

Notes
- MLX requires Apple Silicon + macOS 13+. `mlx-metal` ships the native Metal kernels.
- `mlx_genkit` caches converted models under `./mlx_cache` by default.
- `verifiers` is optional; the current trainer uses the hash/HF/MLX policies directly.

---

## 2. Build / Update the Browsing Environment

Convert `corrected_trajectories.json` (plus any number of raw CSV logs) into graph + task artifacts:

```bash
python scripts/build_env_graph.py \
  --traj corrected_trajectories.json \
  --csv data/my_log.csv \
  --csv-glob "data/*.csv" \
  --csv-max 200 \
  --max_csv_tasks 800 \
  --csv_max_hops 5 \
  --task-config task_filters.json \
  --out env_artifacts
```

Outputs (`env_artifacts/` by default):

- `graph.json` – directed browsing graph (nodes, action-labelled edges, cumulative metadata).
- `tasks_train.json` / `tasks_test.json` – start→goal tasks with reference paths, difficulty and domain tags.
- `csv_sequences.jsonl` – streamed per-participant sequences derived from CSV logs (appends safely across runs).
- `env_config.json` – convenience file used by `scripts/demo_env.py` and `train_gspo.py`.

Incremental updates: re-run the command with `--resume` (and the same cache directory). Already-processed CSVs are skipped; new logs are appended to the sequence file and merged into the graph/task stats.

Task filtering: pass `--task-config path/to/config.json` with keys like:

```json
{
  "include_domains": ["github", "docs"],
  "exclude_hosts": ["x.com"],
  "min_hops": 2,
  "max_hops": 6,
  "sample_rate": 0.5
}
```

See `pliny_env/tasks.py` for the full filter options.

Demo the environment (single random episode, no training):

```bash
python scripts/demo_env.py --cfg env_artifacts/env_config.json
```

Optional observation features (tunable in `EnvConfig` or by editing `env_config.json`):
- `history_window`: number of recent steps to show (default 2).
- `reflection_prompts` + `reflection_every_n`: append a short REFLECTION block (e.g., every step or every Nth step) that invites rationale/alternatives.
- `show_alternatives` + `alternatives_k`: list top‑K outgoing edges at the current node as “ALTERNATIVES (candidates)”.
- `observation_prefix` / `observation_suffix`: free‑form prompt steering before/after the main sections.
- `reward.reward_module` / `reward.reward_class`: plug in a custom reward class (must implement `calculate_step_reward`). Combine with `reward.reward_config` to pass arbitrary parameters.
- `observation_template`: path to a Jinja template (see `pliny_env/obs_templates/default.jinja`) for full control over the observation layout. The default uses the inline format above.

---

## 3. GSPO Training & Policies

`train_gspo.py` unifies the training loop across multiple backends. Each backend implements the shared `SequencePolicy` interface, so swapping engines is entirely CLI-driven.

### Hash baseline (fast smoke test)

```bash
python train_gspo.py \
  --env-config env_artifacts/env_config.json \
  --iterations 10 \
  --group-size 8 \
  --clip-epsilon 0.1 \
  --lr 1e-3 \
  hash
```

### HuggingFace Transformers policy (trainable)

```bash
python train_gspo.py \
  --env-config env_artifacts/env_config.json \
  --iterations 10 \
  --group-size 8 \
  --clip-epsilon 0.1 \
  --lr 1e-3 \
  hf --hf-model sshleifer/tiny-gpt2 --hf-max-new 4
```

Arguments (see `train_gspo.py --help`): temperature, nucleus/top-k sampling, etc. Small open models (`sshleifer/tiny-gpt2`, `distilgpt2`, `Qwen/Qwen2-0.5B-Instruct` via HF) run well on CPU.

### MLX policy via mlx-genkit (inference / evaluation)

```bash
python train_gspo.py \
  --env-config env_artifacts/env_config.json \
  --iterations 1 \
  --group-size 1 \
  mlx --mlx-model Qwen/Qwen2-0.5B-Instruct --mlx-quantize
```

- The policy converts the HF repo once (`./mlx_cache/Qwen_Qwen2-0.5B-Instruct`) and reuses it on subsequent runs.
- GSPO treats MLX runs as evaluation-only (no parameter gradients). Useful for offline scoring or parity checks.

### Remote HTTP / lab endpoints (evaluation)

Many labs expose OpenAI-compatible or custom HTTP endpoints. Use the `http` policy to forward numbered observations to your service and parse the returned decision:

```bash
python train_gspo.py \
  --env-config env_artifacts/env_config.json \
  --iterations 1 \
  --group-size 1 \
  http --http-endpoint https://your-host/api/v1/act \
  --http-headers '{"Authorization": "Bearer <token>"}'
```

Expected JSON response: `{"action_index": <0-based index>, "log_prob": <optional float>}`. Training remains disabled (evaluation only), but this makes it easy to score engines running on vLLM, internal infra, etc.

Trainer internals (`rl/gspo_trainer.py`): sequence-level clipping, group-wise rewards `(r - mean) / std`, optional optimiser (skipped for inference-only policies).

### Quick policy evaluation

```
python scripts/evaluate_policies.py hf --hf-model sshleifer/tiny-gpt2 --episodes 5
python scripts/evaluate_policies.py mlx --mlx-model Qwen/Qwen2-0.5B-Instruct --mlx-quantize --episodes 3
python scripts/evaluate_policies.py http --http-endpoint https://your-host/api/v1/act --episodes 3
```

The script reports average reward/length on train and test splits without performing gradient updates.

Pass `--config policies.json` to benchmark multiple engines; example:

```json
[
  {"name": "hf-tiny", "policy": "hf", "args": {"hf_model": "sshleifer/tiny-gpt2"}},
  {"name": "mlx-qwen", "policy": "mlx", "args": {"mlx_model": "Qwen/Qwen2-0.5B-Instruct", "mlx_quantize": true}},
  {"name": "remote", "policy": "http", "args": {"http_endpoint": "https://your-host/act"}}
]
```

Remote HTTP engines receive the observation, goal URL, and parsed action metadata and should respond with `{"action_index": int, "log_prob": float?}`. This is intended for evaluation/score reporting; GSPO training still requires local gradients (hash/HF/MLX).

Additional flags:
- `--seeds 13,14,15` – evaluate each policy across multiple seeds.
- `--output results.json` – write the consolidated metrics.
- `--csv results.csv` – export aggregate metrics (mean/std reward, lengths) for dashboards.
- `--save-trajectories out_dir` – dump per-policy JSONL trajectories for manual inspection.
- A ready-to-use template lives in `eval_configs/sample_policies.json`.

JSON/CSV summaries include average reward, std, success rate, average episode length, and (when reference paths exist) average path-length ratio relative to the recorded trajectory.

### HTTP Environment Server

Expose the browsing environment over REST so external RL loops can interact without importing this repo:

```bash
python scripts/env_server.py --env-config env_artifacts/env_config.json --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /sessions` – list active sessions.
- `POST /reset` – start a new episode (`{"task_id": optional, "split": "train"|"test"}`) → returns `session_id`, task metadata, initial observation.
- `POST /step` – submit `{ "session_id": ..., "action_index": N }` → returns observation, reward, done flag. Sessions auto-clean up on completion.

The server uses FastAPI. Install dependencies listed above and run via `uvicorn` (built-in when launching the script).

---

## 4. Repository Layout

```
pliny_env/       # graph builders, task generation, env + reward helpers
rl/              # policy interfaces & implementations (hash, HF, MLX), rollouts, trainer utilities
scripts/         # CLI helpers (build env, demo env)
train_gspo.py    # GSPO entry point with policy subcommands
train_grpo.py    # legacy GRPO trainer (MLX sampler integration)
web_browsing_reward.py  # structured reward function used by GRPO pipeline
training/        # checkpoints / exports written by GRPO script
mlx_cache/       # auto-converted MLX models (created on demand)
env_artifacts*/  # example environment builds (train/test splits, graph & sequences)
data/            # structured training/test sets + sample CSVs
```

Key modules

- `pliny_env/graph.py` – merges JSON + CSV logs, supports streaming/incremental builds.
- `pliny_env/tasks.py` – task extraction + filtering (domain/host, hop limits, sampling).
- `pliny_env/env.py` – episodic browsing environment (numbered menus, reward shaping).
- `rl/policy_hash.py` – hash-based policy (PyTorch).
- `rl/policy_hf.py` – HuggingFace policy with prompt → action parsing.
- `rl/policy_mlx.py` – MLX policy using `mlx_genkit` generation + `sequence_logprob`.
- `rl/policy_http.py` – HTTP adapter for external engines (evaluation-only scoring).
- `rl/rollout.py` – sampling episodes, storing log-probs, rewards, reference paths.
- `rl/gspo_trainer.py` – GSPO loss computation, clipping, optional optimiser step.

---

## 5. Data Format (`corrected_trajectories.json` → structured datasets)

Structured training/test data (`data/train_data_structured.json`, `data/test_data_structured.json`) follow this schema:

- `task_id`: unique identifier.
- `prompt`: structured prompt with CONTEXT / THINKING PROCESS / PLANNED ACTIONS sections.
- `completion`: ground-truth completion with step-by-step actions.
- `goal`, `expected_actions`, `expected_outcome`, `difficulty`, `domain`, `format_version`.

The structured JSON files are derived deterministically from `corrected_trajectories.json` (no LLM involvement) and split 40/10 for train/test. Use them directly for supervised fine-tuning baselines or reward modelling.

---

## 6. Legacy GRPO Runner

`train_grpo.py` remains for experimentation with the original MLX sampler + GRPO updates (checkpointing to `training/`). It uses the same environment artifacts generated above but relies on the older reward shaping and sampler-based rollouts.

---

## 7. Caches & Housekeeping

- `mlx_cache/` holds auto-converted MLX models (`auto_load` from `mlx-genkit`). Safe to delete if disk space is needed; they’ll be regenerated on demand.
- `env_artifacts*/` can be regenerated at any time; rerun `scripts/build_env_graph.py` with `--resume` for incremental updates.
- Streaming CSV ingestion writes task sequences to `<out>/csv_sequences.jsonl`; you can tail this file for diagnostics or feed it into other pipelines.
- Use `scripts/cleanup_artifacts.py --dry-run` to preview cache deletions, then rerun without `--dry-run` to reclaim space.
- Explore the data quickly with `scripts/inspect_sequences.py --sequences env_artifacts/csv_sequences.jsonl --graph env_artifacts/graph.json`; it reports sequence counts, average length, and domain/host distributions.

---

## 8. Troubleshooting Tips

- Transformers 4.56+ expects the latest Torch; if you see `torchvision` fake registration errors, uninstall `torchvision`/`torchaudio` or reinstall matching builds.
- MLX imports (`import mlx.core`) require the `libmlx.dylib` packaged with `mlx-metal`. Reinstall via `pip install --force-reinstall mlx mlx-metal` if you hit missing dylib errors.
- Large CSV drops: use `--csv-max` / `--max_csv_tasks` / `--csv_max_hops` to keep artifact sizes manageable.
- Lightweight smoke tests live in `tests/`. Run `pytest` to validate task filters, policy outputs, and environment connectivity.

---

Happy training! With the environment, policy wrappers, and GSPO trainer in place, you can iterate quickly across HF/PT, hash baselines, and MLX parity runs.
