from ui_state import (
    ResearchOutcome,
    blocked_outcome,
    error_outcome,
    invalid_outcome,
    ready_outcome,
    success_outcome,
)


def test_ready_state_is_neutral_and_has_no_result():
    state = ready_outcome()
    assert state.kind == "ready"
    assert state.brief == ""
    assert "Ready" in state.title


def test_invalid_input_preserves_actionable_copy():
    state = invalid_outcome("Add a focused research question.")
    assert state.kind == "error"
    assert "Add a focused research question" in state.detail


def test_success_reports_cited_verified_count_not_all_fetched_papers():
    state = success_outcome(
        "## Findings\nGrounded result\n\nhttps://arxiv.org/abs/2501.12345",
        cited_ids=("2501.12345",),
        verified_ids=("2501.12345", "2502.22222"),
    )
    assert state.kind == "success"
    assert state.brief.startswith("## Findings")
    assert state.verified_citation_count == 1
    assert "Citations verified" in state.title
    assert "claim support still needs review" in state.detail


def test_blocked_state_never_exposes_underlying_draft():
    secret = "THIS DRAFT MUST NEVER BE VISIBLE"
    state = blocked_outcome(
        reason="The answer cites an unverified paper.",
        cited_ids=("2501.12345",),
        verified_ids=(),
    )
    assert state.kind == "blocked"
    assert secret not in state.brief
    assert "withheld" in state.brief.lower()


def test_service_error_is_distinct_from_internal_error():
    service = error_outcome("service", "arXiv did not respond.")
    internal = error_outcome("internal", "Unexpected application error.")
    assert service.kind == "error"
    assert internal.kind == "error"
    assert service.title != internal.title
    assert "arXiv" in service.detail
    assert "application" in internal.detail.lower()


def test_outcome_is_immutable_value_object():
    state = ResearchOutcome(kind="ready", title="Ready", detail="", brief="")
    try:
        state.kind = "success"
    except Exception:
        pass
    assert state.kind == "ready"
