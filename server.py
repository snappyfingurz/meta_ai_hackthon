"""
FastAPI server exposing the EmailTriageEnv via OpenEnv HTTP interface.
"""

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from email_triage_env import Action, EmailTriageEnv


# ── Session store ─────────────────────────────────────────────────────────────

_sessions: dict[str, EmailTriageEnv] = {}


# ── Request / Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "easy"


class StepRequest(BaseModel):
    session_id: str
    action: dict


class StateRequest(BaseModel):
    session_id: str


class CloseRequest(BaseModel):
    session_id: str


class ScoreRequest(BaseModel):
    session_id: str


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OpenEnv — Email Triage",
    description="RL environment for real-world email triage tasks.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "name": "email-triage",
        "version": "1.0.0",
        "tasks": EmailTriageEnv.TASKS,
        "openenv_spec": "1.0",
    }


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.post("/reset")
def reset(req: ResetRequest = Body(default=ResetRequest())):
    session_id = str(uuid.uuid4())
    env = EmailTriageEnv(task_name=req.task_name)
    obs = env.reset()
    _sessions[session_id] = env
    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "task_name": req.task_name,
    }


@app.post("/step")
def step(req: StepRequest):
    env = _get_session(req.session_id)
    try:
        action = Action(**req.action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.post("/state")
def state(req: StateRequest):
    env = _get_session(req.session_id)
    s = env.state()
    data = s.model_dump()
    data["opened_emails"] = list(data.get("opened_emails", []))
    return data


@app.post("/score")
def score(req: ScoreRequest):
    env = _get_session(req.session_id)
    return {"score": env.score()}


@app.post("/close")
def close(req: CloseRequest):
    env = _get_session(req.session_id)
    env.close()
    del _sessions[req.session_id]
    return {"closed": True}


@app.get("/tasks")
def list_tasks():
    from email_triage_env import _load_task
    result = {}
    for t in EmailTriageEnv.TASKS:
        td = _load_task(t)
        result[t] = {
            "description": td["description"],
            "num_emails": len(td["emails"]),
            "max_steps": EmailTriageEnv.MAX_STEPS[t],
        }
    return result


def _get_session(session_id: str) -> EmailTriageEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return env
