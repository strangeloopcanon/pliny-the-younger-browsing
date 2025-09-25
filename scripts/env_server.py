#!/usr/bin/env python3
"""HTTP server exposing PlinyBrowseEnv for external RL loops.

Usage:
  python scripts/env_server.py --env-config env_artifacts/env_config.json

Dependencies:
  pip install fastapi uvicorn
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install fastapi and pydantic: pip install fastapi uvicorn") from exc

from pliny_env.env import EnvConfig, PlinyBrowseEnv

logger = logging.getLogger("env_server")


@dataclass
class Session:
    env: PlinyBrowseEnv
    task: Dict


class SessionManager:
    def __init__(self, env_config: str) -> None:
        self.env_config_path = env_config
        self.env_cfg, self.train_tasks, self.test_tasks = self._load_env_config(env_config)
        self.sessions: Dict[str, Session] = {}

    def _load_env_config(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        env_cfg = EnvConfig(
            graph_path=cfg.get("graph_path", "env_artifacts/graph.json"),
            tasks_path=cfg.get("tasks_path"),
            max_steps=int(cfg.get("max_steps", 10)),
            top_k_actions=int(cfg.get("top_k_actions", 10)),
            include_read_and_stop=cfg.get("include_read_and_stop", True),
            seed=int(cfg.get("seed", 7)),
        )
        tasks_train_path = cfg.get("tasks_path") or "env_artifacts/tasks_train.json"
        tasks_test_path = tasks_train_path.replace("train", "test")
        with open(tasks_train_path, "r", encoding="utf-8") as f:
            train_tasks = json.load(f)
        test_tasks = []
        if tasks_test_path and Path(tasks_test_path).exists():
            with open(tasks_test_path, "r", encoding="utf-8") as f:
                test_tasks = json.load(f)
        return env_cfg, train_tasks, test_tasks

    def create_session(self, task_id: Optional[str] = None, split: str = "train") -> Dict:
        env = PlinyBrowseEnv(self.env_cfg)
        tasks = self.train_tasks if split != "test" else (self.test_tasks or self.train_tasks)
        task = self._select_task(tasks, task_id)
        obs = env.reset(task)
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = Session(env=env, task=task)
        logger.info("Created session %s task %s", session_id, task.get("task_id"))
        return {"session_id": session_id, "task": task, "observation": obs["observation"]}

    def step(self, session_id: str, action_index: int) -> Dict:
        session = self.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        result = session.env.step(action_index)
        if result["done"]:
            self.sessions.pop(session_id, None)
            result["ended"] = True
        return result

    def list_sessions(self) -> Dict[str, Dict]:
        return {
            sid: {
                "task_id": s.task.get("task_id"),
                "current_url": s.env.cur_url,
                "goal_url": s.env.goal_url,
                "step": s.env.step_num,
            }
            for sid, s in self.sessions.items()
        }

    def _select_task(self, tasks, task_id):
        if task_id:
            for task in tasks:
                if task.get("task_id") == task_id:
                    return task
            raise HTTPException(status_code=404, detail="Task ID not found")
        import random

        return random.choice(tasks)


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    split: str = "train"


class StepRequest(BaseModel):
    session_id: str
    action_index: int


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: Dict[str, Optional[str]]
    ended: Optional[bool] = None


class ResetResponse(BaseModel):
    session_id: str
    task: Dict
    observation: str


def create_app(manager: SessionManager) -> FastAPI:
    app = FastAPI(title="Pliny Browsing Env API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/sessions")
    def sessions():
        return manager.list_sessions()

    @app.post("/reset", response_model=ResetResponse)
    def reset(req: ResetRequest):
        return manager.create_session(task_id=req.task_id, split=req.split)

    @app.post("/step", response_model=StepResponse)
    def step(req: StepRequest):
        return manager.step(req.session_id, req.action_index)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP server for PlinyBrowseEnv")
    parser.add_argument("--env-config", default="env_artifacts/env_config.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    manager = SessionManager(args.env_config)
    app = create_app(manager)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
