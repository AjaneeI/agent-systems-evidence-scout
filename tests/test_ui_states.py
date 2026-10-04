import app


def test_state_helpers_render_product_states():
    assert "Evidence contract satisfied" in app._status_html("success", "Evidence contract satisfied", "2 papers verified.")
    assert "status-card blocked" in app._status_html("blocked", "Draft withheld", "Citation failed.")
    assert "status-card error" in app._status_html("error", "Service unavailable", "Retry.")
    activity = app._activity_html([("done", "Verified paper"), ("blocked", "Gate blocked")])
    assert "Research activity" in activity
    assert 'class="done"' in activity
    assert 'class="blocked"' in activity
