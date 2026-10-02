import app


class _CapturingCodeAgent:
    captured_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).captured_kwargs = kwargs


def test_code_agent_authorizes_json_import(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(app, "InferenceClientModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(app, "CodeAgent", _CapturingCodeAgent)

    app.build_agent()

    authorized = _CapturingCodeAgent.captured_kwargs.get("additional_authorized_imports")
    assert authorized is not None
    assert "json" in authorized
