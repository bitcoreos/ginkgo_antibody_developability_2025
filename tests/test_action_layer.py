import importlib
import os


def test_action_layer_gather_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("TMP_DIR", str(tmp_path))

    cfg_mod = importlib.import_module("workspace.config")
    importlib.reload(cfg_mod)

    a = importlib.import_module("workspace.bitcore_research.action_layer")
    importlib.reload(a)

    engine = a.ActionEngine()
    res = engine.gather_data("test query", "general")

    assert res["query"] == "test query"
    assert "summary" in res and "total_sources" in res["summary"]

    files = [p.name for p in tmp_path.iterdir() if p.name.startswith("data_collection_")]
    assert files, "expected dry-run artifact in TMP_DIR"
