import os

from workspace.config import cfg


def test_run_initialization_dry_run(tmp_path):
    # Ensure DRY_RUN is enabled for tests
    cfg.DRY_RUN = True
    cfg.KNOWLEDGE_BASE_PATH = str(tmp_path / "kb.json")
    cfg.ASSURANCE_REPORTS_DIR = str(tmp_path / "assurance")
    cfg.TMP_DIR = str(tmp_path / "tmp")

    from workspace.bitcore_research.init import run_initialization

    results = run_initialization()

    assert 'perception' in results
    assert 'reasoning' in results
    assert 'action' in results
    assert 'reflection' in results
    assert 'summary' in results
    # summary should include success_count and component_count
    assert results['summary']['component_count'] == 4
