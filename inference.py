"""
inference.py — OpenEnv RL Challenge submission script for Email Triage.

Reads env vars, connects to the environment server via OpenEnv HTTP API,
and runs an LLM agent (via OpenAI client) through all three tasks.

Output format (stdout):
    [START] task=<task> env=email-triage model=<model>
    [STEP]  step=<n> action=<action_str> reward=<r> done=<bool> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import json
import os
import sys
import traceback
from typing import Any

import requests
from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI client ─────────────────────────────────────────────────────────────

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert email triage assistant. Your job is to process an inbox and:
1. Open each email to read it
2. Categorize each email as: urgent | normal | spam
3. Set a priority order (most urgent first)
4. Draft replies to emails that need responses
5. Escalate emails that require senior attention (legal, security, large financial decisions, HR complaints)
6. Call done() when finished

You MUST respond with a single JSON object representing ONE action at a time.

Action schemas:
  {"action_type": "open",       "email_id": "e1"}
  {"action_type": "categorize", "email_id": "e1", "category": "urgent|normal|spam"}
  {"action_type": "prioritize", "email_ids": ["e1","e2","e3"]}
  {"action_type": "draft_reply","email_id": "e1", "subject": "RE: ...", "body": "..."}
  {"action_type": "escalate",   "email_id": "e1", "reason": "Requires legal review"}
  {"action_type": "done"}

Rules:
- Open an email BEFORE categorizing or replying to it
- Escalate emails about: security breaches, legal matters, large wire transfers, HR complaints
- Spam emails never need replies or escalation
- Reply to urgent emails and actionable normal emails
- After opening all emails and completing all actions, call done()

Respond with ONLY the JSON object, no explanation.
""".strip()


# ── Environment API helpers ───────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict) -> dict:
    url = f"{ENV_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def reset_env(task_name: str) -> tuple[str, dict]:
    data = api_post("reset", {"task_name": task_name})
    return data["session_id"], data["observation"]


def step_env(session_id: str, action: dict) -> tuple[dict, float, bool, dict]:
    data = api_post("step", {"session_id": session_id, "action": action})
    return data["observation"], data["reward"], data["done"], data["info"]


def close_env(session_id: str) -> None:
    try:
        api_post("close", {"session_id": session_id})
    except Exception:
        pass


def get_score(session_id: str) -> float:
    try:
        data = api_post("score", {"session_id": session_id})
        return data.get("score", 0.01)
    except Exception:
        return 0.01


# ── LLM agent ────────────────────────────────────────────────────────────────

def build_user_message(obs: dict) -> str:
    lines = [f"Step {obs['step_number']} | {obs['message']}", ""]
    lines.append("=== INBOX ===")
    for email in obs.get("inbox", []):
        lines.append(
            f"[{email['id']}] From: {email['sender']} | "
            f"Subject: {email['subject']} | "
            f"Time: {email['timestamp']}"
        )
    if obs.get("current_email"):
        e = obs["current_email"]
        lines.append(f"\n=== OPEN EMAIL [{e['id']}] ===")
        lines.append(f"From: {e['sender']}")
        lines.append(f"Subject: {e['subject']}")
        lines.append(f"Body: {e['body']}")
    if obs.get("actions_taken"):
        a = obs["actions_taken"][-1]
        lines.append("\n=== ACTIONS SO FAR ===")
        lines.append(f"Categorized: {a.get('categorizations', {})}")
        lines.append(f"Priority:    {a.get('priority_order', [])}")
        lines.append(f"Drafted:     {a.get('drafts', [])}")
        lines.append(f"Escalated:   {a.get('escalations', [])}")
    return "\n".join(lines)


def get_llm_action(conversation: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation,
        max_tokens=512,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_name: str) -> dict:
    session_id = None
    rewards: list[float] = []
    steps = 0
    success = False
    last_error = None

    print(f"[START] task={task_name} env=email-triage model={MODEL_NAME}", flush=True)

    try:
        session_id, obs = reset_env(task_name)
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        while True:
            user_msg = build_user_message(obs)
            conversation.append({"role": "user", "content": user_msg})

            try:
                action = get_llm_action(conversation)
            except Exception as exc:
                last_error = f"LLM parse error: {exc}"
                action = {"action_type": "done"}

            action_str = json.dumps(action)
            obs, reward, done, info = step_env(session_id, action)
            steps += 1
            rewards.append(reward)
            last_error = info.get("error")

            conversation.append({
                "role": "assistant",
                "content": action_str,
            })

            reward_fmt = f"{reward:.2f}"
            done_str   = "true" if done else "false"
            error_str  = last_error if last_error else "null"
            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward_fmt} done={done_str} error={error_str}",
                flush=True,
            )

            if done:
                score = get_score(session_id)
                success = score >= 0.6
                break

    except Exception as exc:
        last_error = str(exc)
        traceback.print_exc(file=sys.stderr)
    finally:
        if session_id:
            close_env(session_id)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )
    return {"task": task_name, "success": success, "steps": steps, "rewards": rewards}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tasks = os.getenv("TASKS", "easy,medium,hard").split(",")
    results = []
    for task in tasks:
        task = task.strip()
        if task:
            result = run_episode(task)
            results.append(result)

    # Summary to stderr (not stdout, to keep stdout clean)
    print("\n=== SUMMARY ===", file=sys.stderr)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(
            f"  {status} {r['task']:8s}  steps={r['steps']:3d}  "
            f"total_reward={sum(r['rewards']):.2f}",
            file=sys.stderr,
        )
