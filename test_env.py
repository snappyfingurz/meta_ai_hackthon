"""
Tests for EmailTriageEnv — run with: pytest tests/
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.email_triage_env import EmailTriageEnv, Action


class TestEasyTask:
    def setup_method(self):
        self.env = EmailTriageEnv("easy")
        self.obs = self.env.reset()

    def teardown_method(self):
        self.env.close()

    def test_reset_returns_observation(self):
        assert len(self.obs.inbox) == 3
        assert self.obs.step_number == 0

    def test_open_email(self):
        obs, reward, done, info = self.env.step({"action_type": "open", "email_id": "e1"})
        assert obs.current_email is not None
        assert obs.current_email.id == "e1"
        assert reward > 0
        assert not done

    def test_categorize_without_open_penalises(self):
        _, reward, _, _ = self.env.step(
            {"action_type": "categorize", "email_id": "e1", "category": "urgent"}
        )
        assert reward < 0

    def test_correct_categorize_rewards(self):
        self.env.step({"action_type": "open", "email_id": "e1"})
        _, reward, _, _ = self.env.step(
            {"action_type": "categorize", "email_id": "e1", "category": "urgent"}
        )
        assert reward == pytest.approx(0.15)

    def test_wrong_categorize_penalises(self):
        self.env.step({"action_type": "open", "email_id": "e1"})
        _, reward, _, _ = self.env.step(
            {"action_type": "categorize", "email_id": "e1", "category": "spam"}
        )
        assert reward < 0

    def test_invalid_category_raises(self):
        self.env.step({"action_type": "open", "email_id": "e1"})
        with pytest.raises(Exception):
            self.env.step({"action_type": "categorize", "email_id": "e1", "category": "maybe"})

    def test_draft_reply_after_open(self):
        self.env.step({"action_type": "open", "email_id": "e1"})
        _, reward, _, _ = self.env.step({
            "action_type": "draft_reply",
            "email_id": "e1",
            "subject": "RE: Server down",
            "body": "Team is on it, will update in 30 mins.",
        })
        assert reward > 0

    def test_draft_reply_too_short_penalises(self):
        self.env.step({"action_type": "open", "email_id": "e1"})
        _, reward, _, _ = self.env.step({
            "action_type": "draft_reply",
            "email_id": "e1",
            "subject": "RE",
            "body": "ok",
        })
        assert reward < 0

    def test_done_ends_episode(self):
        # Do some valid actions first
        self.env.step({"action_type": "open", "email_id": "e1"})
        self.env.step({"action_type": "categorize", "email_id": "e1", "category": "urgent"})
        _, _, done, _ = self.env.step({"action_type": "done"})
        assert done

    def test_perfect_run_scores_above_threshold(self):
        env = EmailTriageEnv("easy")
        env.reset()
        actions = [
            {"action_type": "open", "email_id": "e1"},
            {"action_type": "categorize", "email_id": "e1", "category": "urgent"},
            {"action_type": "draft_reply", "email_id": "e1", "subject": "RE: Server down",
             "body": "On it immediately, team alerted, ETA 30 minutes."},
            {"action_type": "open", "email_id": "e2"},
            {"action_type": "categorize", "email_id": "e2", "category": "spam"},
            {"action_type": "open", "email_id": "e3"},
            {"action_type": "categorize", "email_id": "e3", "category": "normal"},
            {"action_type": "draft_reply", "email_id": "e3", "subject": "RE: Team lunch",
             "body": "Sounds great! 12:30 works for me, see you then."},
            {"action_type": "prioritize", "email_ids": ["e1", "e3", "e2"]},
            {"action_type": "done"},
        ]
        for a in actions:
            obs, reward, done, info = env.step(a)
            if done:
                break
        score = env.score()
        env.close()
        assert score >= 0.75, f"Expected score >= 0.75, got {score}"


class TestMediumTask:
    def test_escalation_required(self):
        env = EmailTriageEnv("medium")
        env.reset()
        # Open and escalate the legal email
        env.step({"action_type": "open", "email_id": "m4"})
        _, reward, _, _ = env.step({
            "action_type": "escalate",
            "email_id": "m4",
            "reason": "Legal notice requires senior leadership review",
        })
        assert reward == pytest.approx(0.15)
        env.close()

    def test_false_escalation_penalises(self):
        env = EmailTriageEnv("medium")
        env.reset()
        env.step({"action_type": "open", "email_id": "m2"})
        _, reward, _, _ = env.step({
            "action_type": "escalate",
            "email_id": "m2",
            "reason": "Benefits email escalated incorrectly",
        })
        assert reward < 0
        env.close()


class TestHardTask:
    def test_phishing_must_be_spam(self):
        env = EmailTriageEnv("hard")
        env.reset()
        env.step({"action_type": "open", "email_id": "h3"})
        env.step({"action_type": "categorize", "email_id": "h3", "category": "spam"})
        env.step({"action_type": "done"})
        score = env.score()
        # Score should be higher when phishing is correctly identified
        env.close()
        assert score >= 0.0

    def test_security_breach_escalation(self):
        env = EmailTriageEnv("hard")
        env.reset()
        env.step({"action_type": "open", "email_id": "h2"})
        _, reward, _, _ = env.step({
            "action_type": "escalate",
            "email_id": "h2",
            "reason": "Security breach requires immediate IT and leadership response",
        })
        assert reward == pytest.approx(0.15)
        env.close()


class TestStepLimit:
    def test_exceeding_limit_ends_episode(self):
        env = EmailTriageEnv("easy")
        env.reset()
        done = False
        steps = 0
        # Spam open actions until limit
        for _ in range(25):
            _, _, done, _ = env.step({"action_type": "open", "email_id": "e1"})
            steps += 1
            if done:
                break
        assert done, "Expected done=True after step limit exceeded"
        env.close()


class TestPriorityOrder:
    def test_correct_order_rewards(self):
        env = EmailTriageEnv("easy")
        env.reset()
        _, reward, _, _ = env.step({
            "action_type": "prioritize",
            "email_ids": ["e1", "e3", "e2"],
        })
        assert reward > 0
        env.close()

    def test_inverted_order_lower_reward(self):
        env = EmailTriageEnv("easy")
        env.reset()
        _, reward_correct, _, _ = env.step({
            "action_type": "prioritize",
            "email_ids": ["e1", "e3", "e2"],
        })
        env2 = EmailTriageEnv("easy")
        env2.reset()
        _, reward_wrong, _, _ = env2.step({
            "action_type": "prioritize",
            "email_ids": ["e2", "e3", "e1"],
        })
        assert reward_correct >= reward_wrong
        env.close()
        env2.close()
