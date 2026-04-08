---
title: Email Triage RL Environment
emoji: 📬
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: latest
pinned: false
---

# 📬 Email Triage — OpenEnv RL Environment

A real-world reinforcement learning environment where an AI agent must intelligently triage an email inbox: categorizing messages, setting priorities, drafting replies, and escalating critical items (security breaches, legal notices, HR complaints, large financial decisions).

---

## Motivation

Email triage is a high-value, daily task for knowledge workers. It requires reasoning about urgency, context, sender intent, and risk — making it an ideal RL benchmark that is:

- **Real-world**: mirrors actual inbox management
- **Multi-step**: requires planning across many actions
- **Reward-dense**: incremental feedback on each correct decision
- **Scalable**: 3 difficulty levels, extensible to new task packs

---

## Environment Overview

### Observation Space

| Field | Type | Description |
|---|---|---|
| `inbox` | `list[Email]` | All emails (id, sender, subject, body, timestamp) |
| `current_email` | `Email \| null` | Currently open email |
| `actions_taken` | `list[dict]` | Summary of categorizations, drafts, escalations |
| `step_number` | `int` | Current step count |
| `message` | `str` | Human-readable feedback from last action |

### Action Space

| Action | Parameters | Description |
|---|---|---|
| `open` | `email_id` | Open an email to read its body |
| `categorize` | `email_id`, `category` | Label as `urgent`, `normal`, or `spam` |
| `prioritize` | `email_ids` | Set ordered list (most urgent first) |
| `draft_reply` | `email_id`, `subject`, `body` | Compose a reply |
| `escalate` | `email_id`, `reason` | Flag for human escalation |
| `done` | — | Signal task completion |

### Reward Function

Rewards are **dense** — provided after every action:

| Action | Reward |
|---|---|
| Open new email | +0.02 |
| Correct categorization | +0.15 |
| Wrong categorization | −0.05 |
| Categorize unread email | −0.05 |
| Correct escalation | +0.15 |
| False escalation | −0.05 |
| Valid draft reply | +0.03 to +0.10 |
| Draft reply too short | −0.05 |
| Correct priority order | +0.05 to +0.10 |
| Exceeding step limit | −0.05/step |
| Terminal (done) bonus | proportional to final score + efficiency |

---

## Tasks

### Easy (`task_name=easy`)
- **Emails**: 3
- **Max steps**: 20
- **Objective**: Identify the urgent server-down alert, ignore the marketing spam, reply to the colleague's lunch invite.
- **Difficulty**: ⭐

### Medium (`task_name=medium`)
- **Emails**: 6
- **Max steps**: 35
- **Objective**: Triage 6 emails including a legal notice that must be escalated to leadership. Distinguish the phishing-style spam from legitimate vendor mail.
- **Difficulty**: ⭐⭐

### Hard (`task_name=hard`)
- **Emails**: 9
- **Max steps**: 50
- **Objective**: Handle a security breach alert, a phishing attempt (must NOT be escalated), an HR harassment complaint, a $45K wire transfer approval, and a partner deadline. Multiple escalations required.
- **Difficulty**: ⭐⭐⭐

---

## Grading

Each task is scored 0.0–1.0 based on a weighted combination:

| Criterion | Easy | Medium | Hard |
|---|---|---|---|
| Categorization accuracy | 50% | 35% | 30% |
| Priority order | 30% | 25% | 20% |
| Reply drafts | 20% | 20% | 20% |
| Escalation decisions | — | 20% | 25% |
| Phishing bonus/penalty | — | — | ±5% |

---

## Baseline Scores

Measured with GPT-4.1-mini (zero-shot):

| Task | Score |
|---|---|
| easy | 0.82 |
| medium | 0.61 |
| hard | 0.44 |

---

## Setup & Usage

### Local (Docker)

```bash
# Build
docker build -t email-triage .

# Run environment server
docker run -p 7860:7860 email-triage

# In another terminal — run inference
export HF_TOKEN=your_api_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Local (Python)

```bash
pip install -r requirements.txt

# Start server
uvicorn server:app --port 7860

# Run inference
export HF_TOKEN=your_key
python inference.py
```

### API Quick Start

```python
import requests

# 1. Start a session
resp = requests.post("http://localhost:7860/reset", json={"task_name": "easy"})
session_id = resp.json()["session_id"]
obs = resp.json()["observation"]

# 2. Open first email
resp = requests.post("http://localhost:7860/step", json={
    "session_id": session_id,
    "action": {"action_type": "open", "email_id": "e1"}
})

# 3. Categorize it
resp = requests.post("http://localhost:7860/step", json={
    "session_id": session_id,
    "action": {"action_type": "categorize", "email_id": "e1", "category": "urgent"}
})

# 4. Get final score
score = requests.post("http://localhost:7860/score", json={"session_id": session_id}).json()

# 5. Close session
requests.post("http://localhost:7860/close", json={"session_id": session_id})
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Environment info |
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks with metadata |
| POST | `/reset` | Start new episode |
| POST | `/step` | Execute action |
| POST | `/state` | Get current state |
| POST | `/score` | Get final score |
| POST | `/close` | End session |

---

## Project Structure

```
.
├── inference.py          # Hackathon submission script (OpenAI client agent)
├── server.py             # FastAPI HTTP server (OpenEnv interface)
├── openenv.yaml          # Environment metadata
├── requirements.txt
├── Dockerfile
├── README.md
└── env/
    └── email_triage_env.py   # Core environment + tasks + graders
```

---

## Hardware Requirements

Runs comfortably within 2 vCPU / 8 GB RAM. The environment itself is stateless per session and uses no ML models server-side — all inference is done client-side via the OpenAI API.
