"""Dependency-free regressions for the broken automated file edit."""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def test_application_is_complete_and_syntactically_valid():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"build_agent", "run_research", "build_demo"} <= functions
    assert any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in tree.body)


def test_research_remains_string_returning_and_guarded():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    legacy = functions["run_research"]
    outcome = functions["run_research_outcome"]

    assert ast.unparse(legacy.returns) == "str"
    legacy_calls = [ast.unparse(n.func) for n in ast.walk(legacy) if isinstance(n, ast.Call)]
    outcome_calls = [ast.unparse(n.func) for n in ast.walk(outcome) if isinstance(n, ast.Call)]

    assert "run_research_outcome" in legacy_calls
    assert "validate_final_answer" in outcome_calls
    assert "reset_verification_registry" in outcome_calls
    texts = [n.value for n in ast.walk(outcome) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "\n\nResearch question:\n" in texts
