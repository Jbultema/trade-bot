from types import SimpleNamespace

from trade_bot import system_identity


def test_native_identity_exposes_config_drift_without_selecting_a_winner(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    for file in ("baseline.yaml", "active_trading.yaml"):
        (tmp_path / "configs" / file).write_text("test: true\n")
    monkeypatch.setattr(system_identity, "load_config", lambda path: SimpleNamespace(primary_strategy=path.name))
    value = system_identity.identity(tmp_path)
    assert not value["configuration_consistent"]
    assert value["primary"] == "baseline.yaml"
    assert value["operating_primary"] == "active_trading.yaml"
    assert not value["action_authorized"]
