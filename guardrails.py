from __future__ import annotations

import re
from dataclasses import dataclass

from tools import get_verified_arxiv_ids, normalize_arxiv_id

_ARXIV_URL_OR_ID_RE = re.compile(
    r"(?:https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/)?"
    r"((?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.\-]+/\d{7})(?:v\d+)?)"
    r"(?:\.pdf)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    reason: str
    cited_ids: tuple[str, ...]
    verified_ids: tuple[str, ...]


def extract_arxiv_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for match in _ARXIV_URL_OR_ID_RE.finditer(text or ""):
        try:
            normalized = normalize_arxiv_id(match.group(1))
        except ValueError:
            continue
        ids.add(re.sub(r"v\d+$", "", normalized, flags=re.IGNORECASE))
    return ids


def validate_final_answer(answer: str) -> GuardrailResult:
    if not isinstance(answer, str) or not answer.strip():
        return GuardrailResult(False, "The agent produced an empty answer.", tuple(), tuple())

    cited = extract_arxiv_ids(answer)
    verified = {re.sub(r"v\d+$", "", item, flags=re.IGNORECASE) for item in get_verified_arxiv_ids()}

    if not cited:
        return GuardrailResult(
            False,
            "The answer contains no arXiv citation, so its research claims are not auditable.",
            tuple(),
            tuple(sorted(verified)),
        )

    unverified = cited - verified
    if unverified:
        return GuardrailResult(
            False,
            "The answer cites papers that were not verified during this run: "
            + ", ".join(sorted(unverified)),
            tuple(sorted(cited)),
            tuple(sorted(verified)),
        )

    return GuardrailResult(
        True,
        "All cited arXiv papers were verified during this run.",
        tuple(sorted(cited)),
        tuple(sorted(verified)),
    )


def enforce_final_answer(answer: str) -> str:
    result = validate_final_answer(answer)
    if result.passed:
        return answer

    return (
        "GUARDRAIL BLOCKED THE DRAFT\n\n"
        f"Reason: {result.reason}\n\n"
        "The underlying draft is intentionally withheld because the evidence contract was not satisfied. "
        "Run the research task again and verify every cited arXiv paper before presenting conclusions."
    )
