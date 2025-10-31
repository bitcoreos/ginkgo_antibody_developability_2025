import json
from pathlib import Path

import pytest

from workspace.bitcore_research import launch_research_swarm as swarm_module
from workspace.bitcore_research import validate_research_findings as validation_module


def _configure_swarm_cfg(tmp_path, monkeypatch):
    base = tmp_path / "results"
    raw = base / "raw_results"
    base.mkdir()
    raw.mkdir()

    monkeypatch.setattr(swarm_module.cfg, "RESULTS_BASE_DIR", str(base), raising=False)
    monkeypatch.setattr(swarm_module.cfg, "RESULTS_RAW_DIR", str(raw), raising=False)
    monkeypatch.setattr(swarm_module.cfg, "SWARM_MAX_WORKERS", 2, raising=False)
    monkeypatch.setattr(swarm_module.cfg, "MORPHEUS_API_URL", "https://example.com", raising=False)
    monkeypatch.setattr(swarm_module.cfg, "MORPHEUS_API_KEY", "test", raising=False)

    return base, raw


def test_launch_research_swarm_dry_run_writes_results(tmp_path, monkeypatch):
    _, raw = _configure_swarm_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(swarm_module, "is_dry_run", lambda: True)

    def _unexpected_post(*args, **kwargs):  # pragma: no cover - safety net
        raise AssertionError("requests.post should not be called in dry-run")

    monkeypatch.setattr(swarm_module.requests, "post", _unexpected_post)

    payload = {"researchTopic": "Test topic"}
    models = [{"model": "model-a", "system_prompt": "prompt"}]

    results = swarm_module.launch_research_swarm(payload=payload, models=models, max_workers=1)

    assert results["model-a"]["status"] == "success"
    assert results["model-a"]["response"]["simulated"] is True

    saved_files = list(Path(raw).glob("research_results_*.json"))
    assert len(saved_files) == 1


def test_launch_research_swarm_handles_request_error(tmp_path, monkeypatch):
    _, raw = _configure_swarm_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(swarm_module, "is_dry_run", lambda: False)

    class DummyResponse:
        status_code = 500
        text = "failure"

        def json(self):  # pragma: no cover - not used in this branch
            return {"error": "failure"}

    monkeypatch.setattr(swarm_module.requests, "post", lambda *args, **kwargs: DummyResponse())

    results = swarm_module.launch_research_swarm(
        payload={"researchTopic": "Test"},
        models=[{"model": "model-b", "system_prompt": "prompt"}],
        max_workers=1,
    )

    assert results["model-b"]["status"] == "error"
    assert results["model-b"]["status_code"] == 500

    saved_files = list(Path(raw).glob("research_results_*.json"))
    assert saved_files, "Expected a results file to be written"


def _configure_validation_cfg(tmp_path, monkeypatch):
    base = tmp_path / "results"
    raw = base / "raw_results"
    logs = base / "logs"
    raw.mkdir(parents=True)
    logs.mkdir(parents=True)

    monkeypatch.setattr(validation_module.cfg, "RESULTS_BASE_DIR", str(base), raising=False)
    monkeypatch.setattr(validation_module.cfg, "RESULTS_RAW_DIR", str(raw), raising=False)
    monkeypatch.setattr(validation_module.cfg, "LOG_FILE", str(logs / "validation.log"), raising=False)

    return base, raw


def test_analyze_convergence_creates_validated_file(tmp_path, monkeypatch):
    base, raw = _configure_validation_cfg(tmp_path, monkeypatch)

    data = {
        "results": {
            "model-a": {
                "status": "success",
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "Hidden pattern in antibody sequences improves developability via HMM models."
                            }
                        }
                    ]
                },
            },
            "model-b": {
                "status": "success",
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "Hidden pattern in antibody sequences improves developability via HMM models."
                            }
                        }
                    ]
                },
            },
            "model-c": {
                "status": "error",
                "message": "timeout",
            },
        }
    }

    results_file = raw / "research_results_test.json"
    results_file.write_text(json.dumps(data))

    output_path = validation_module.analyze_convergence(str(results_file))
    assert output_path is not None

    output = json.loads(Path(output_path).read_text())
    assert output["total_convergent_findings"] == 1
    assert output["findings"][0]["confidence"] == "low"
    assert output["findings"][0]["model_count"] == 2


def test_run_validation_returns_none_when_no_files(tmp_path, monkeypatch):
    base, raw = _configure_validation_cfg(tmp_path, monkeypatch)

    assert validation_module.run_validation() is None

    # Create a file so run_validation finds it
    data = {
        "results": {
            "model-a": {
                "status": "success",
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "Hidden pattern in antibody sequences improves developability via HMM models."
                            }
                        }
                    ]
                },
            },
            "model-b": {
                "status": "success",
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "Hidden pattern in antibody sequences improves developability via HMM models."
                            }
                        }
                    ]
                },
            },
        }
    }

    results_file = raw / "research_results_20250101.json"
    results_file.write_text(json.dumps(data))

    output_path = validation_module.run_validation()
    assert output_path is not None
    assert Path(output_path).exists()