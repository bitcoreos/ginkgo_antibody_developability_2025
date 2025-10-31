import importlib
import os


def test_config_defaults(monkeypatch, tmp_path):
    for key in [
        "DRY_RUN",
        "MORPHEUS_API_URL",
        "RESULTS_BASE_DIR",
        "RESULTS_RAW_DIR",
        "KNOWLEDGE_BASE_PATH",
        "ASSURANCE_REPORTS_DIR",
        "TMP_DIR",
    ]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("RESULTS_BASE_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("RESULTS_RAW_DIR", str(tmp_path / "results" / "raw"))
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", str(tmp_path / "kb.json"))
    monkeypatch.setenv("ASSURANCE_REPORTS_DIR", str(tmp_path / "assurance"))
    monkeypatch.setenv("TMP_DIR", str(tmp_path / "tmp"))

    cfg_mod = importlib.import_module("workspace.config")
    importlib.reload(cfg_mod)
    cfg = cfg_mod.cfg

    assert cfg.DRY_RUN is True
    assert cfg.MORPHEUS_API_URL.startswith("https://")
    assert cfg.TMP_DIR.endswith("tmp")
    assert cfg.SWARM_MAX_WORKERS == 6
