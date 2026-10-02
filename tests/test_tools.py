import json

import pytest

import tools


def fake_record(arxiv_id: str):
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "authors": ["A. Researcher"],
        "published": "2026-01-01T00:00:00+00:00",
        "updated": "2026-01-01T00:00:00+00:00",
        "abstract": "Evidence about agent systems.",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "primary_category": "cs.AI",
        "categories": ["cs.AI"],
    }


def test_normalize_arxiv_id_accepts_id():
    assert tools.normalize_arxiv_id("2601.12345") == "2601.12345"


def test_normalize_arxiv_id_accepts_abs_url():
    assert tools.normalize_arxiv_id("https://arxiv.org/abs/2601.12345v2") == "2601.12345v2"


def test_normalize_arxiv_id_rejects_non_arxiv_url():
    with pytest.raises(ValueError):
        tools.normalize_arxiv_id("https://example.com/2601.12345")


def test_search_arxiv_caps_results_before_network():
    with pytest.raises(ValueError):
        tools.search_arxiv("agent systems", 6)


def test_verify_registers_paper(monkeypatch):
    tools.reset_verification_registry()
    monkeypatch.setattr(tools, "_fetch_paper_by_id", lambda _: fake_record("2601.12345v2"))
    payload = tools.verify_arxiv_paper("2601.12345")
    assert '"verified": true' in payload
    assert tools.get_verified_arxiv_ids() == {"2601.12345"}


def test_compare_requires_two_unique_papers():
    with pytest.raises(ValueError):
        tools.compare_arxiv_papers("2601.12345")


def test_compare_verifies_all_papers(monkeypatch):
    tools.reset_verification_registry()
    monkeypatch.setattr(
        tools,
        "_fetch_paper_by_id",
        lambda x: fake_record(tools.normalize_arxiv_id(x)),
    )
    payload = tools.compare_arxiv_papers(
        "2601.12345, 2602.54321",
        "reliability",
    )
    assert '"paper_count": 2' in payload
    assert tools.get_verified_arxiv_ids() == {"2601.12345", "2602.54321"}


def test_verify_accepts_arxiv_id_keyword(monkeypatch):
    tools.reset_verification_registry()
    monkeypatch.setattr(tools, "_fetch_paper_by_id", lambda _: fake_record("2601.12345v2"))
    payload = tools.verify_arxiv_paper(arxiv_id="2601.12345")
    assert '"verified": true' in payload
    assert tools.get_verified_arxiv_ids() == {"2601.12345"}


def test_verify_accepts_abs_url_keyword(monkeypatch):
    tools.reset_verification_registry()
    monkeypatch.setattr(
        tools,
        "_fetch_paper_by_id",
        lambda x: fake_record(tools.normalize_arxiv_id(x)),
    )
    payload = tools.verify_arxiv_paper(arxiv_id="https://arxiv.org/abs/2601.12345v2")
    assert '"verified": true' in payload
    assert '"arxiv_id": "2601.12345v2"' in payload
    assert tools.get_verified_arxiv_ids() == {"2601.12345"}


def test_compare_accepts_abs_urls_and_versioned_ids(monkeypatch):
    tools.reset_verification_registry()
    monkeypatch.setattr(
        tools,
        "_fetch_paper_by_id",
        lambda x: fake_record(tools.normalize_arxiv_id(x)),
    )
    payload = tools.compare_arxiv_papers(
        "https://arxiv.org/abs/2601.12345v2, 2602.54321v1",
        "reliability",
    )
    assert '"paper_count": 2' in payload
    assert tools.get_verified_arxiv_ids() == {"2601.12345", "2602.54321"}


def test_compare_accepts_list_of_ids(monkeypatch):
    tools.reset_verification_registry()
    monkeypatch.setattr(
        tools,
        "_fetch_paper_by_id",
        lambda x: fake_record(tools.normalize_arxiv_id(x)),
    )
    payload = tools.compare_arxiv_papers(["2601.12345", "2602.54321"])
    assert '"paper_count": 2' in payload
    assert tools.get_verified_arxiv_ids() == {"2601.12345", "2602.54321"}


def test_compare_rejects_malformed_id():
    with pytest.raises(ValueError, match="canonical arXiv IDs"):
        tools.compare_arxiv_papers("2601.12345, not-a-paper")


def test_tool_outputs_parse_as_json_and_yield_canonical_ids(monkeypatch):
    tools.reset_verification_registry()
    monkeypatch.setattr(
        tools,
        "_fetch_paper_by_id",
        lambda x: fake_record(tools.normalize_arxiv_id(x)),
    )
    verify_payload = json.loads(tools.verify_arxiv_paper(arxiv_id="2601.12345"))
    verified_id = verify_payload["paper"]["arxiv_id"]

    compare_payload = json.loads(
        tools.compare_arxiv_papers(f"{verified_id}, 2602.54321", "reliability")
    )
    compared_ids = [paper["arxiv_id"] for paper in compare_payload["papers"]]
    assert compared_ids == ["2601.12345", "2602.54321"]
    for arxiv_id in compared_ids:
        assert tools.normalize_arxiv_id(arxiv_id) == arxiv_id
    assert tools.get_verified_arxiv_ids() == {"2601.12345", "2602.54321"}
