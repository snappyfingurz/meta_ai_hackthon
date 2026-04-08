"""
Email Triage Environment — OpenEnv-compliant implementation.

Simulates a real-world email inbox where an agent must triage emails by:
- Categorizing (urgent/normal/spam)
- Prioritizing order of response
- Drafting reply subjects
- Flagging emails needing escalation
"""

import json
import re
import time
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field


# ── Observation / Action / State models ──────────────────────────────────────

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    has_attachment: bool = False


class Observation(BaseModel):
    inbox: list[Email]
    current_email: Email | None = None
    actions_taken: list[dict] = Field(default_factory=list)
    step_number: int = 0
    message: str = ""


class Action(BaseModel):
    """
    Supported action types:
      open(email_id)                          – open an email for reading
      categorize(email_id, category)          – category: urgent|normal|spam
      prioritize(email_ids)                   – ordered list, most urgent first
      draft_reply(email_id, subject, body)    – draft a reply
      escalate(email_id, reason)              – flag for human escalation
      done()                                  – signal task completion
    """
    action_type: str
    email_id: str | None = None
    category: str | None = None
    email_ids: list[str] | None = None
    subject: str | None = None
    body: str | None = None
    reason: str | None = None


class TaskState(BaseModel):
    task_name: str
    emails: list[Email]
    categorizations: dict[str, str] = Field(default_factory=dict)
    priority_order: list[str] = Field(default_factory=list)
    drafts: dict[str, dict] = Field(default_factory=dict)
    escalations: dict[str, str] = Field(default_factory=dict)
    current_email_id: str | None = None
    steps: int = 0
    done: bool = False
    opened_emails: set[str] = Field(default_factory=set)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context):
        # opened_emails may deserialize as list; convert to set
        if isinstance(self.opened_emails, list):
            self.opened_emails = set(self.opened_emails)


# ── Reward helpers ────────────────────────────────────────────────────────────

CATEGORY_VALUES = {"urgent": 0, "normal": 1, "spam": 2}
VALID_CATEGORIES = set(CATEGORY_VALUES.keys())


def _partial_category_reward(categorizations: dict, expected: dict) -> float:
    if not expected:
        return 0.0
    correct = sum(1 for eid, cat in categorizations.items()
                  if expected.get(eid) == cat)
    return correct / len(expected)


def _priority_order_reward(priority_order: list, expected_order: list) -> float:
    """Kendall-tau-like: fraction of pairs in correct relative order."""
    if len(expected_order) < 2:
        return 1.0 if priority_order == expected_order else 0.0
    ids = [e for e in expected_order if e in priority_order]
    if len(ids) < 2:
        return 0.0
    pairs_correct = 0
    total_pairs = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if a in priority_order and b in priority_order:
                total_pairs += 1
                if priority_order.index(a) < priority_order.index(b):
                    pairs_correct += 1
    return pairs_correct / total_pairs if total_pairs else 0.0


# ── Main Environment ──────────────────────────────────────────────────────────

class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage environment.

    Usage:
        env = EmailTriageEnv(task_name="easy")
        obs = env.reset()
        obs, reward, done, info = env.step(action_dict)
        env.close()
    """

    TASKS = ["easy", "medium", "hard"]
    MAX_STEPS = {"easy": 20, "medium": 35, "hard": 50}

    def __init__(self, task_name: str = "easy"):
        if task_name not in self.TASKS:
            raise ValueError(f"task_name must be one of {self.TASKS}")
        self.task_name = task_name
        self._state: TaskState | None = None
        self._task_def: dict = {}
        self._start_time = None

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self._task_def = _load_task(self.task_name)
        self._state = TaskState(
            task_name=self.task_name,
            emails=self._task_def["emails"],
        )
        self._start_time = time.time()
        return self._build_obs("Inbox loaded. Begin triaging emails.")

    def step(self, action: dict | Action) -> tuple[Observation, float, bool, dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        if isinstance(action, dict):
            action = Action(**action)

        self._state.steps += 1
        reward = 0.0
        info: dict[str, Any] = {"error": None}

        # — Loop / step-limit penalty ——————————————————————————————————————————
        max_steps = self.MAX_STEPS[self.task_name]
        if self._state.steps > max_steps:
            reward -= 0.05
            info["error"] = f"Exceeded max steps ({max_steps})"
            return self._build_obs("Step limit exceeded."), reward, True, info

        try:
            reward, message = self._dispatch(action, info)
        except Exception as exc:
            info["error"] = str(exc)
            reward = -0.1
            message = f"Error: {exc}"

        done = self._state.done
        obs = self._build_obs(message)
        return obs, reward, done, info

    def state(self) -> TaskState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return deepcopy(self._state)

    def close(self):
        self._state = None

    def score(self) -> float:
        """Final task score [0, 1]. Call after done=True."""
        if self._state is None:
            return 0.0
        grader = _get_grader(self.task_name)
        return grader(self._state, self._task_def)

    # ── Action dispatch ───────────────────────────────────────────────────────

    def _dispatch(self, action: Action, info: dict) -> tuple[float, str]:
        at = action.action_type.lower()

        if at == "open":
            return self._act_open(action, info)
        elif at == "categorize":
            return self._act_categorize(action, info)
        elif at == "prioritize":
            return self._act_prioritize(action, info)
        elif at == "draft_reply":
            return self._act_draft_reply(action, info)
        elif at == "escalate":
            return self._act_escalate(action, info)
        elif at == "done":
            return self._act_done(info)
        else:
            raise ValueError(f"Unknown action_type: {action.action_type}")

    def _act_open(self, action: Action, info: dict) -> tuple[float, str]:
        eid = action.email_id
        email = self._get_email(eid)
        self._state.current_email_id = eid
        already_opened = eid in self._state.opened_emails
        self._state.opened_emails.add(eid)
        reward = 0.0 if already_opened else 0.02
        return reward, f"Opened email {eid}: '{email.subject}'"

    def _act_categorize(self, action: Action, info: dict) -> tuple[float, str]:
        eid = action.email_id
        cat = (action.category or "").lower()
        self._get_email(eid)  # validate exists
        if cat not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {VALID_CATEGORIES}")
        if eid not in self._state.opened_emails:
            return -0.05, f"Cannot categorize unread email {eid}"

        expected = self._task_def.get("expected_categories", {})
        prev = self._state.categorizations.get(eid)
        self._state.categorizations[eid] = cat

        if expected and eid in expected:
            if expected[eid] == cat:
                reward = 0.15 if prev != cat else 0.0
                return reward, f"Categorized {eid} as '{cat}' ✓"
            else:
                return -0.05, f"Categorized {eid} as '{cat}' (expected: {expected[eid]})"
        return 0.05, f"Categorized {eid} as '{cat}'"

    def _act_prioritize(self, action: Action, info: dict) -> tuple[float, str]:
        ids = action.email_ids or []
        known = {e.id for e in self._state.emails}
        bad = [i for i in ids if i not in known]
        if bad:
            raise ValueError(f"Unknown email ids: {bad}")
        self._state.priority_order = ids
        expected = self._task_def.get("expected_priority", [])
        reward = 0.1 * _priority_order_reward(ids, expected) if expected else 0.05
        return reward, f"Priority set: {ids}"

    def _act_draft_reply(self, action: Action, info: dict) -> tuple[float, str]:
        eid = action.email_id
        self._get_email(eid)
        if eid not in self._state.opened_emails:
            return -0.05, f"Cannot reply to unread email {eid}"
        subj = action.subject or ""
        body = action.body or ""
        if len(body.strip()) < 10:
            return -0.05, "Reply body too short (< 10 chars)"
        self._state.drafts[eid] = {"subject": subj, "body": body}
        expected_replies = self._task_def.get("expected_reply_ids", [])
        reward = 0.1 if eid in expected_replies else 0.03
        return reward, f"Draft saved for {eid}"

    def _act_escalate(self, action: Action, info: dict) -> tuple[float, str]:
        eid = action.email_id
        self._get_email(eid)
        reason = (action.reason or "").strip()
        if len(reason) < 5:
            return -0.02, "Escalation reason too brief"
        self._state.escalations[eid] = reason
        expected_esc = self._task_def.get("expected_escalations", [])
        reward = 0.15 if eid in expected_esc else -0.05
        return reward, f"Escalated {eid}: {reason}"

    def _act_done(self, info: dict) -> tuple[float, str]:
        grader = _get_grader(self.task_name)
        final_score = grader(self._state, self._task_def)
        self._state.done = True
        bonus = max(0.0, 0.1 * (1 - self._state.steps / self.MAX_STEPS[self.task_name]))
        reward = final_score * 0.5 + bonus
        return reward, f"Task complete. Final score: {final_score:.2f}"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_email(self, eid: str) -> Email:
        for e in self._state.emails:
            if e.id == eid:
                return e
        raise ValueError(f"No email with id '{eid}'")

    def _build_obs(self, message: str) -> Observation:
        current = None
        if self._state and self._state.current_email_id:
            for e in self._state.emails:
                if e.id == self._state.current_email_id:
                    current = e
                    break
        actions_taken = []
        if self._state:
            actions_taken = [
                {"categorizations": self._state.categorizations,
                 "priority_order": self._state.priority_order,
                 "drafts": list(self._state.drafts.keys()),
                 "escalations": list(self._state.escalations.keys())}
            ]
        return Observation(
            inbox=self._state.emails if self._state else [],
            current_email=current,
            actions_taken=actions_taken,
            step_number=self._state.steps if self._state else 0,
            message=message,
        )


# ── Task definitions ──────────────────────────────────────────────────────────

def _load_task(task_name: str) -> dict:
    tasks = {
        "easy": _task_easy(),
        "medium": _task_medium(),
        "hard": _task_hard(),
    }
    return tasks[task_name]


def _task_easy() -> dict:
    """3 emails, clear categories, no escalation needed."""
    emails = [
        Email(id="e1", sender="boss@company.com", subject="URGENT: Server down",
              body="Our main server is down! Clients are calling. Fix ASAP.",
              timestamp="2024-01-15T09:00:00Z"),
        Email(id="e2", sender="newsletter@deals.com", subject="50% off everything today!",
              body="Flash sale! Click here for amazing deals on everything in our store.",
              timestamp="2024-01-15T08:30:00Z"),
        Email(id="e3", sender="colleague@company.com", subject="Team lunch tomorrow",
              body="Hey, are you free for lunch tomorrow? Thinking 12:30 at the usual place.",
              timestamp="2024-01-15T08:00:00Z"),
    ]
    return {
        "emails": emails,
        "expected_categories": {"e1": "urgent", "e2": "spam", "e3": "normal"},
        "expected_priority": ["e1", "e3", "e2"],
        "expected_reply_ids": ["e1", "e3"],
        "expected_escalations": [],
        "description": "Triage 3 emails: identify the urgent server issue, ignore spam, respond to colleague.",
    }


def _task_medium() -> dict:
    """6 emails, mixed categories, one escalation required."""
    emails = [
        Email(id="m1", sender="ceo@bigclient.com", subject="Contract renewal — decision needed",
              body="We need a decision on the contract renewal by EOD Friday. "
                   "If we don't hear back we will go with another vendor.",
              timestamp="2024-01-15T09:15:00Z"),
        Email(id="m2", sender="hr@company.com", subject="Benefits enrollment closes Friday",
              body="Reminder: open enrollment for health benefits ends this Friday. "
                   "Log in to the HR portal to make your selections.",
              timestamp="2024-01-15T08:45:00Z"),
        Email(id="m3", sender="unknown1928@temp.xyz", subject="You have won $10,000!!!",
              body="Congratulations! You've been selected to receive $10,000. "
                   "Click here and provide your bank details to claim your prize.",
              timestamp="2024-01-15T08:00:00Z"),
        Email(id="m4", sender="legal@company.com", subject="Lawsuit notice — confidential",
              body="We have received a legal notice from a former employee. "
                   "This requires immediate attention from senior leadership and legal counsel. "
                   "Do not discuss via email.",
              timestamp="2024-01-15T09:00:00Z"),
        Email(id="m5", sender="dev@company.com", subject="PR review request",
              body="Hi, could you review my pull request when you get a chance? "
                   "No rush — it's for next sprint.",
              timestamp="2024-01-15T07:30:00Z"),
        Email(id="m6", sender="support@saas-tool.com", subject="Your invoice is ready",
              body="Your monthly invoice for $299 is ready. Download it from the billing portal.",
              timestamp="2024-01-15T07:00:00Z"),
    ]
    return {
        "emails": emails,
        "expected_categories": {
            "m1": "urgent", "m2": "normal", "m3": "spam",
            "m4": "urgent", "m5": "normal", "m6": "normal",
        },
        "expected_priority": ["m1", "m4", "m2", "m5", "m6", "m3"],
        "expected_reply_ids": ["m1", "m2"],
        "expected_escalations": ["m4"],
        "description": "Triage 6 emails including a legal notice that must be escalated.",
    }


def _task_hard() -> dict:
    """9 emails, ambiguous senders, mixed signals, multiple escalations."""
    emails = [
        Email(id="h1", sender="cto@partner-corp.io", subject="RE: Integration deadline",
              body="Following up on our call — we need the API keys and sandbox access today "
                   "or we push the launch to Q2. This affects $2M in projected revenue.",
              timestamp="2024-01-15T09:30:00Z"),
        Email(id="h2", sender="security@company.com", subject="⚠️ Breach detected — action required",
              body="Our SIEM detected anomalous login activity from IP 203.0.113.45. "
                   "Three admin accounts were accessed at 03:00 UTC. "
                   "Recommend immediate password reset and audit.",
              timestamp="2024-01-15T09:25:00Z"),
        Email(id="h3", sender="no-reply@phishing-sim.net", subject="Security alert: verify your account",
              body="Unusual activity detected on your account. Click here to verify: "
                   "http://company-login.phishing-sim.net/verify",
              timestamp="2024-01-15T09:20:00Z"),
        Email(id="h4", sender="employee123@company.com", subject="Confidential: HR complaint",
              body="I need to report a serious workplace harassment issue involving my direct manager. "
                   "I'm afraid of retaliation. Please advise on next steps confidentially.",
              timestamp="2024-01-15T09:00:00Z"),
        Email(id="h5", sender="finance@company.com", subject="Wire transfer approval needed — TODAY",
              body="The vendor payment of $45,000 to Acme Supplies must be approved by 3pm "
                   "or we incur late fees. PO#: 98123. Please reply to approve.",
              timestamp="2024-01-15T08:50:00Z"),
        Email(id="h6", sender="marketing@company.com", subject="Blog post draft for review",
              body="Hey! Attached is the draft for next week's blog post. "
                   "Let me know if you have feedback by Wednesday.",
              timestamp="2024-01-15T08:30:00Z"),
        Email(id="h7", sender="vendor@supplies.biz", subject="New product catalog 2024",
              body="Please find our updated 2024 catalog. We have new items you might be interested in. "
                   "Reply for a quote.",
              timestamp="2024-01-15T08:00:00Z"),
        Email(id="h8", sender="it@company.com", subject="Scheduled maintenance tonight 11pm-2am",
              body="Reminder: systems will be offline for maintenance tonight from 11pm to 2am EST. "
                   "Please save your work and log off before 10:45pm.",
              timestamp="2024-01-15T07:30:00Z"),
        Email(id="h9", sender="ceo@company.com", subject="All-hands next week — RSVP",
              body="We're holding a company-wide all-hands next Thursday at 2pm. "
                   "Please RSVP by EOD Monday. Attendance is expected.",
              timestamp="2024-01-15T07:00:00Z"),
    ]
    return {
        "emails": emails,
        "expected_categories": {
            "h1": "urgent", "h2": "urgent", "h3": "spam",
            "h4": "urgent", "h5": "urgent", "h6": "normal",
            "h7": "normal",  "h8": "normal", "h9": "normal",
        },
        "expected_priority": ["h2", "h1", "h4", "h5", "h9", "h8", "h6", "h7", "h3"],
        "expected_reply_ids": ["h1", "h5", "h9"],
        "expected_escalations": ["h2", "h4", "h5"],
        "description": (
            "Triage 9 emails: security breach, phishing, harassment complaint, large wire transfer, "
            "partner deadline. Multiple escalations required; distinguish phishing from real security alert."
        ),
    }


# ── Graders ───────────────────────────────────────────────────────────────────

def _get_grader(task_name: str):
    return {
        "easy": _grade_easy,
        "medium": _grade_medium,
        "hard": _grade_hard,
    }[task_name]


def _clamp_score(score: float) -> float:
    """Clamp score to strictly between 0 and 1, excluding exact 0.0 and 1.0."""
    epsilon = 0.01
    return max(epsilon, min(1.0 - epsilon, round(score, 4)))


def _grade_easy(state: TaskState, task_def: dict) -> float:
    cat_score = _partial_category_reward(state.categorizations, task_def["expected_categories"])
    pri_score = _priority_order_reward(state.priority_order, task_def["expected_priority"])
    reply_ids = set(state.drafts.keys())
    reply_score = len(reply_ids & {"e1", "e3"}) / 2
    score = cat_score * 0.5 + pri_score * 0.3 + reply_score * 0.2
    return _clamp_score(score)


def _grade_medium(state: TaskState, task_def: dict) -> float:
    cat_score = _partial_category_reward(state.categorizations, task_def["expected_categories"])
    pri_score = _priority_order_reward(state.priority_order, task_def["expected_priority"])
    reply_score = len(set(state.drafts.keys()) & {"m1", "m2"}) / 2
    escalation_score = 1.0 if "m4" in state.escalations else 0.0
    # Penalise false escalations
    false_esc = len([e for e in state.escalations if e != "m4"]) * 0.1
    escalation_score = max(0.0, escalation_score - false_esc)
    score = cat_score * 0.35 + pri_score * 0.25 + reply_score * 0.20 + escalation_score * 0.20
    return _clamp_score(score)


def _grade_hard(state: TaskState, task_def: dict) -> float:
    cat_score = _partial_category_reward(state.categorizations, task_def["expected_categories"])
    pri_score = _priority_order_reward(state.priority_order, task_def["expected_priority"])
    expected_replies = {"h1", "h5", "h9"}
    reply_score = len(set(state.drafts.keys()) & expected_replies) / len(expected_replies)
    expected_esc = {"h2", "h4", "h5"}
    correct_esc = len(set(state.escalations.keys()) & expected_esc)
    false_esc = len([e for e in state.escalations if e not in expected_esc])
    escalation_score = max(0.0, correct_esc / len(expected_esc) - false_esc * 0.15)
    # Phishing detection bonus: h3 must be spam, NOT escalated
    phishing_ok = state.categorizations.get("h3") == "spam" and "h3" not in state.escalations
    phishing_bonus = 0.05 if phishing_ok else -0.05
    raw = (
        cat_score * 0.30
        + pri_score * 0.20
        + reply_score * 0.20
        + escalation_score * 0.25
        + phishing_bonus
    )
    return _clamp_score(raw)
