# Agent Systems Evidence Scout

An evidence-controlled arXiv research agent that lets an LLM choose research tools while deterministic Python enforces the sourcing contract.

**Stack:** Python · smolagents · Hugging Face Inference · arXiv · Gradio · pytest

> Built as a CS 697 applied-AI project, extending the arXiv agent pattern demonstrated in Inal Mashukov's `UMBInal/arxiv-agent-lab` reference implementation. The reference architecture is credited below; the verification tools, deterministic citation gate, tests, integration hardening, and Evidence Scout workflow are this project's extension.

## The problem

Research agents can produce fluent answers while still citing papers they never actually verified. Agent Systems Evidence Scout separates **model discretion** from **evidence enforcement**:

- the agent can decide what to search and which research tools to call;
- verification tools resolve only canonical arXiv IDs / URLs;
- the final answer is checked by deterministic Python outside the LLM;
- if a cited paper was not verified during that run, the draft is withheld.

That boundary is the core design decision: **the model can choose tools, but it cannot waive its own sourcing requirement.**

## Workflow

```text
Research question
      |
      v
  CodeAgent
      |
      +--> search_arxiv
      |
      +--> verify_arxiv_paper
      |
      +--> compare_arxiv_papers
      |
      v
  evidence synthesis
      |
      v
deterministic citation gate
      |
      +--> PASS: verified evidence brief
      |
      +--> FAIL: withhold draft + explain failure
```

The user receives a compact brief with:

- **Findings**
- **Evidence**
- **Uncertainty / limitations**
- **Verified references**

## Reliability controls

| Control | Behavior |
| --- | --- |
| Canonical paper validation | Verification accepts only canonical arXiv IDs or `arxiv.org` abs/pdf URLs |
| Bounded retrieval | Search returns at most 5 papers |
| Bounded comparison | Comparison accepts 2–5 unique papers |
| Untrusted retrieval content | Retrieved titles and abstracts are treated as research data, not instructions |
| Per-run verification registry | Papers must be verified in the same run before they can support the final response |
| Deterministic final gate | A draft with no auditable arXiv citation or an unverified citation is withheld |
| Secret handling | `HF_TOKEN` is read from the environment; `.env` is ignored |

## Evaluation

The current implementation has been checked at two levels:

- **19 deterministic tests passed** on the development Mac.
- `python -m py_compile app.py tools.py guardrails.py` passed.
- The deterministic suite avoids live model/arXiv calls so control logic is repeatable.
- A live Hugging Face + arXiv end-to-end run completed successfully in **4 CodeAgent steps**.
- The agent's ceiling remained **6 steps**; the successful run did not require increasing the budget.

### Why the live test mattered

The first live run exposed an integration failure chain that the original offline suite did not catch.

1. The model naturally called `verify_arxiv_paper(arxiv_id=...)`, but the tool's parameter name did not match that interface.
2. Tool results were JSON strings, but `json` was not authorized inside the CodeAgent interpreter.
3. `eval` remained correctly blocked. Forced into manual string parsing, the model malformed a comparison input.
4. Those failed calls consumed the six-step budget and the run stopped.

The fix was deliberately small:

- align the verification tool with the natural `arxiv_id=` call;
- authorize only the standard-library `json` import;
- let the comparison tool accept list/tuple input while preserving canonical-ID validation;
- add regression tests for the failure path.

**I did not solve the failure by giving the agent more steps.** After fixing the interfaces, the same live scenario completed in four.

That distinction is useful in agent engineering: deterministic tests validate controls, while live model/tool execution can expose interface behavior that unit tests miss.

## Core tools

### `search_arxiv`

Runs a bounded relevance search and returns canonical metadata and abstracts as JSON.

### `verify_arxiv_paper`

Resolves an arXiv ID or arXiv URL, checks that arXiv returned the requested paper, and records successful verification for the current run.

### `compare_arxiv_papers`

Builds a bounded evidence bundle for two to five verified papers so the agent can compare only retrieved evidence.

### Final citation guardrail

`guardrails.py` extracts arXiv citations from the final draft and compares them with the per-run verification registry. If the evidence contract fails, the underlying draft is intentionally withheld.

## Run locally

Requires Python and a Hugging Face token with access to the configured inference model.

```bash
git clone https://github.com/AjaneeI/agent-systems-evidence-scout.git
cd agent-systems-evidence-scout

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN="your_token_here"
python app.py
```

Open the local Gradio URL printed in the terminal.

## Test

```bash
pytest -q
python -m py_compile app.py tools.py guardrails.py
```

## Example research prompts

- What recent arXiv work evaluates reliability or failure modes in LLM agents?
- Compare two or more papers on human oversight or guardrails for agentic AI.
- What evidence exists on when multi-agent systems outperform simpler single-agent approaches?

The last question was used to exercise the live integration path; its answer is **demo content, not a claim that multi-agent systems generally outperform single-agent systems**.

## What this project demonstrates

- designing tool interfaces for model-driven execution;
- separating probabilistic reasoning from deterministic policy enforcement;
- building testable guardrails around external evidence;
- validating agent systems with both offline tests and live execution;
- diagnosing cascading tool/interface failures instead of masking them with larger execution budgets;
- translating a research workflow into a small, inspectable AI application.

## Scope and limitations

This is a compact applied-AI project, not a production research platform.

- Evidence is based on arXiv metadata and abstracts rather than full-paper analysis.
- The verification registry is in-memory and scoped to a single run.
- There is no production authentication, persistence, tracing platform, or hosted deployment.
- A successful live run validates the workflow, not the truth of every research conclusion.
- `compare_arxiv_papers` is covered by deterministic tests; an agent may choose individual verification instead during a particular live trajectory.

## Project provenance

This project began in **CS 697 — Industry AI Applications Lab** and uses Inal Mashukov's `UMBInal/arxiv-agent-lab` Hugging Face Space as a reference architecture for the compact smolagents + Hugging Face + arXiv + Gradio pattern.

Reference implementation: https://huggingface.co/spaces/UMBInal/arxiv-agent-lab

The public repository contains the standalone project extension only; unrelated coursework and the private class-project repository remain private.
