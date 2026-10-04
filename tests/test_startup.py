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
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_research")
    assert ast.unparse(run.returns) == "str"
    calls = [ast.unparse(n.func) for n in ast.walk(run) if isinstance(n, ast.Call)]
    assert "enforce_final_answer" in calls
    assert "reset_verification_registry" in calls
    texts = [n.value for n in ast.walk(run) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "\n\nResearch question:\n" in texts
