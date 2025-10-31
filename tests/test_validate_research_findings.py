import importlib
import json
import os


def test_analyze_convergence_end_to_end(tmp_path, monkeypatch):
    # Create fake results structure
    results = {
        "payload": {"researchTopic": "test"},
        "models": ["m1", "m2"],
        "results": {
            "m1": {"status": "success", "response": {"choices": [{"message": {"content": "Hidden motif A found in sequence"}}]}},
            "m2": {"status": "success", "response": {"choices": [{"message": {"content": "Hidden motif A also appears in sequence"}}]}}
        }
    }

    fn = tmp_path / "fake_results.json"
    with open(fn, 'w') as f:
        json.dump(results, f)

    monkeypatch.setenv("RESULTS_BASE_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("RESULTS_RAW_DIR", str(tmp_path / "results" / "raw"))
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", str(tmp_path / "kb.json"))
    monkeypatch.setenv("ASSURANCE_REPORTS_DIR", str(tmp_path / "assurance"))
    monkeypatch.setenv("TMP_DIR", str(tmp_path / "tmp"))

    cfg_mod = importlib.import_module("workspace.config")
    importlib.reload(cfg_mod)

    mod = importlib.import_module("workspace.bitcore_research.validate_research_findings")
    importlib.reload(mod)

    out = mod.analyze_convergence(str(fn))
    assert out is None or os.path.exists(out)
