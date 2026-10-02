import tools
from guardrails import enforce_final_answer, extract_arxiv_ids, validate_final_answer


def test_extracts_arxiv_urls_and_ids():
    text = "See https://arxiv.org/abs/2601.12345 and 2602.54321v2."
    assert extract_arxiv_ids(text) == {"2601.12345", "2602.54321"}


def test_blocks_answer_without_citations():
    tools.reset_verification_registry()
    result = validate_final_answer("A claim with no auditable paper reference.")
    assert not result.passed
    assert "no arXiv citation" in result.reason


def test_blocks_unverified_citation():
    tools.reset_verification_registry()
    answer = "Evidence: https://arxiv.org/abs/2601.12345"
    result = validate_final_answer(answer)
    assert not result.passed
    assert "not verified" in result.reason


def test_allows_only_verified_citations(monkeypatch):
    monkeypatch.setattr("guardrails.get_verified_arxiv_ids", lambda: {"2601.12345"})
    answer = "Evidence: https://arxiv.org/abs/2601.12345"
    result = validate_final_answer(answer)
    assert result.passed


def test_enforcement_withholds_failed_draft():
    tools.reset_verification_registry()
    guarded = enforce_final_answer("Unsupported https://arxiv.org/abs/2601.12345")
    assert guarded.startswith("GUARDRAIL BLOCKED THE DRAFT")
    assert "Unsupported" not in guarded
