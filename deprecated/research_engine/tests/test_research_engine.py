import importlib
import json


def _reload_config(monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_BASE_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("RESULTS_RAW_DIR", str(tmp_path / "results" / "raw"))
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", str(tmp_path / "kb.json"))
    monkeypatch.setenv("ASSURANCE_REPORTS_DIR", str(tmp_path / "assurance"))
    monkeypatch.setenv("TMP_DIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("DRY_RUN", "true")

    cfg_mod = importlib.import_module("workspace.config")
    importlib.reload(cfg_mod)
    return cfg_mod


def test_dispatch_to_neurons_dry_run(monkeypatch, tmp_path):
    _reload_config(monkeypatch, tmp_path)

    engine = importlib.import_module("workspace.research_engine.engine")
    importlib.reload(engine)

    results = engine.dispatch_to_neurons(
        topic="dry run antibodies",
        system_prompt="You are a dry-run tester.",
    )

    neurons = engine.load_default_neurons()
    assert set(results.keys()) == {neuron.name for neuron in neurons}
    for payload in results.values():
        assert payload["status"] == "success"
        assert payload["response"].get("simulated") is True


def test_save_results(monkeypatch, tmp_path):
    cfg_mod = _reload_config(monkeypatch, tmp_path)
    engine = importlib.import_module("workspace.research_engine.engine")
    importlib.reload(engine)

    results = {
        "n1": {"status": "success", "response": {"simulated": True}},
    }

    path = engine.save_results(
        topic="antibodies",
        system_prompt="Test",
        results=results,
        output_dir=str(tmp_path / "out"),
        prefix="unit_test",
    )

    with open(path, "r", encoding="utf-8") as handle:
        content = json.load(handle)

    assert content["topic"] == "antibodies"
    assert content["dry_run"] is True
    assert content["results"]["n1"]["status"] == "success"


def test_cli_dry_run(monkeypatch, tmp_path):
    _reload_config(monkeypatch, tmp_path)

    cli = importlib.import_module("workspace.research_engine.cli")
    importlib.reload(cli)

    exit_code = cli.main([
        "--topic",
        "cli dry run",
        "--system-prompt",
        "Test CLI",
        "--no-save",
        "--dry-run",
    ])

    assert exit_code == 0