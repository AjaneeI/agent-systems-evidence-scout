import app
import tools


class FakeAgent:
    def __init__(self, answer, verified_ids=()):
        self.answer = answer
        self.verified_ids = verified_ids

    def run(self, prompt):
        assert "Research question:" in prompt
        tools._VERIFIED_PAPERS.update(self.verified_ids)
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
    monkeypatch.setattr(app, "build_agent", lambda: FakeAgent(
        "## Findings\nSupported finding.\n\n## Verified references\nhttps://arxiv.org/abs/2501.12345",
        verified_ids=("2501.12345",),
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


def test_begin_research_clears_stale_success_before_new_run():
    brief, status = app.begin_research()
    assert "Researching" in brief
    assert "Researching" in status
    assert "Citations verified" not in status


def test_structured_agent_result_is_rendered_as_markdown():
    raw = {
        "Findings": "Multi-agent systems can outperform simpler approaches in bounded settings.",
        "Evidence": [
            "Paper A: https://arxiv.org/abs/2106.06828v1",
            "Paper B: https://arxiv.org/abs/2610.00538v1",
        ],
        "Uncertainty / limitations": "The evidence is domain-specific.",
        "Verified references": [
            "https://arxiv.org/abs/2106.06828v1",
            "https://arxiv.org/abs/2610.00538v1",
        ],
    }
    rendered = app.normalize_agent_output(raw)
    assert rendered.startswith("## Findings")
    assert "## Evidence" in rendered
    assert "## Uncertainty / limitations" in rendered
    assert "## Verified references" in rendered
    assert "{'Findings':" not in rendered


def test_stringified_mapping_is_normalized_without_eval_execution():
    raw = (
        "{'Findings': 'A finding', "
        "'Evidence': ['https://arxiv.org/abs/2501.12345'], "
        "'Uncertainty / limitations': 'Limited scope', "
        "'Verified references': ['https://arxiv.org/abs/2501.12345']}"
    )
    rendered = app.normalize_agent_output(raw)
    assert rendered.startswith("## Findings")
    assert "- https://arxiv.org/abs/2501.12345" in rendered
    assert "{'Findings':" not in rendered


def test_plain_markdown_agent_result_is_preserved():
    raw = "## Findings\nAlready formatted.\n\n## Verified references\nhttps://arxiv.org/abs/2501.12345"
    assert app.normalize_agent_output(raw) == raw


def test_markdown_preamble_is_removed_before_findings():
    raw = (
        "Based on the provided comparison variable, here is the answer:\n\n"
        "## Findings\nUseful finding.\n\n"
        "## Evidence\nEvidence body.\n\n"
        "## Uncertainty / limitations\nLimited scope.\n\n"
        "## Verified references\nhttps://arxiv.org/abs/2501.12345"
    )
    rendered = app.normalize_agent_output(raw)
    assert rendered.startswith("## Findings")
    assert "comparison variable" not in rendered
