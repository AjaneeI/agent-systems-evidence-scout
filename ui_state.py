from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchOutcome:
    kind: str
    title: str
    detail: str
    brief: str
    verified_citation_count: int = 0


def ready_outcome() -> ResearchOutcome:
    return ResearchOutcome(
        kind="ready",
        title="Ready to research",
        detail="Ask a focused question to begin.",
        brief="",
    )


def invalid_outcome(detail: str) -> ResearchOutcome:
    return ResearchOutcome(
        kind="error",
        title="Check your research question",
        detail=detail,
        brief="",
    )


def success_outcome(
    brief: str,
    *,
    cited_ids: tuple[str, ...],
    verified_ids: tuple[str, ...],
) -> ResearchOutcome:
    verified = set(verified_ids)
    count = len(set(cited_ids) & verified)
    return ResearchOutcome(
        kind="success",
        title="Citations verified",
        detail=(
            f"{count} cited paper identit{'y' if count == 1 else 'ies'} checked against arXiv; "
            "claim support still needs review."
        ),
        brief=brief,
        verified_citation_count=count,
    )


def blocked_outcome(
    *,
    reason: str,
    cited_ids: tuple[str, ...],
    verified_ids: tuple[str, ...],
) -> ResearchOutcome:
    del cited_ids, verified_ids
    return ResearchOutcome(
        kind="blocked",
        title="Draft withheld",
        detail=reason,
        brief=(
            "## Evidence contract not satisfied\n\n"
            "The draft was withheld because its citation requirements were not satisfied. "
            "Run the research task again before presenting conclusions."
        ),
    )


def error_outcome(category: str, detail: str) -> ResearchOutcome:
    titles = {
        "service": "Research service unavailable",
        "token": "Hugging Face token required",
        "internal": "Application error",
    }
    return ResearchOutcome(
        kind="error",
        title=titles.get(category, "Research could not complete"),
        detail=detail,
        brief="",
    )
