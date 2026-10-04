import app
import tools


class FakeAgent:
    def __init__(self, answer):
        self.answer = answer

    def run(self, prompt):
        assert "Research question:" in prompt
        return self.answer


def test_missing_token_is_actionable(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    state = app.run_research_outcome("What evidence exists?")
    assert state.kind == "error"
    assert state.title == "Hugging Face token required"
    assert "HF_TOKEN" in state.detail


def test_verified_citation_returns_success_without_overclaim(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    tools.reset_verification_registry()
    tools._VERIFIED_PAPERS.add("2501.12345")
    monkeypatch.setattr(app, "build_agent", lambda: FakeAgent(
        "## Findings\nSupported finding.\n\n## Verified references\nhttps://arxiv.org/abs/2501.12345"
    ))
    state = app.run_research_outcome("What evidence exists?")
    assert state.kind == "success"
    assert state.verified_citation_count == 1
    assert "claim support still needs review" in state.detail


def test_unverified_citation_withholds_draft(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    tools.reset_verification_registry()
    secret = "SECRET DRAFT CLAIM"
    monkeypatch.setattr(app, "build_agent", lambda: FakeAgent(
        f"## Findings\n{secret}\n\nhttps://arxiv.org/abs/2501.12345"
    ))
    state = app.run_research_outcome("What evidence exists?")
    assert state.kind == "blocked"
    assert secret not in state.brief


def test_consecutive_run_cannot_inherit_previous_verification(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    tools._VERIFIED_PAPERS.add("2501.12345")
    monkeypatch.setattr(app, "build_agent", lambda: FakeAgent(
        "## Findings\nClaim\n\nhttps://arxiv.org/abs/2501.12345"
    ))
    state = app.run_research_outcome("New question")
    assert state.kind == "blocked"


def test_unexpected_programming_error_is_not_called_service_outage(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    def boom():
        raise TypeError("programming bug")
    monkeypatch.setattr(app, "build_agent", boom)
    state = app.run_research_outcome("Question")
    assert state.kind == "error"
    assert state.title == "Application error"
    assert "programming bug" not in state.detail
