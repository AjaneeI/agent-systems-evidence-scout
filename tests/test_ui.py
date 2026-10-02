import app


def test_public_ui_uses_blocks_and_portfolio_copy():
    demo = app.build_demo()

    assert demo.__class__.__name__ == "Blocks"
    assert demo.title == "Agent Systems Evidence Scout"
    assert "#07111f" in app.CSS
    assert "#39d8c3" in app.CSS


def test_public_ui_preserves_agent_limits():
    assert app.MAX_STEPS == 6
    assert "verify every paper" in app.SYSTEM_TASK.lower()
