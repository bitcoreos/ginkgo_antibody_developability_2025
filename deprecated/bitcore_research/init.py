
#!/usr/bin/env python3
"""
Safe initializer for the Autonomous Research Coordinator.

Provides a `run_initialization()` function that performs dry-run safe checks of
the perception, reasoning, action, and reflection components. Does not execute
on import. Uses `workspace.config.cfg` for paths and DRY_RUN behavior.
"""

import os
from datetime import datetime

from workspace.config import cfg, get_logger, is_dry_run

logger = get_logger(__name__)


def _ensure_dirs():
    # Create required runtime directories where possible
    for path in [os.path.dirname(cfg.LOG_FILE), cfg.TMP_DIR]:
        try:
            if path:
                os.makedirs(path, exist_ok=True)
        except Exception:
            pass


def run_initialization():
    """Run component smoke-tests in DRY_RUN-safe mode.

    Returns a dict with component results.
    """
    _ensure_dirs()

    results = {}

    # Perception
    try:
        from workspace.bitcore_research import perception_layer as perception
        logger.info("Testing perception layer connection (DRY_RUN=%s)", cfg.DRY_RUN)
        res = perception.test_connection()
        results['perception'] = {'ok': res.get('success', False), 'detail': res}
    except Exception as e:
        logger.error("Perception layer test failed: %s", e)
        results['perception'] = {'ok': False, 'error': str(e)}

    # Reasoning
    try:
        from workspace.bitcore_research import reasoning_layer as reasoning
        logger.info("Testing reasoning layer")
        engine = reasoning.ReasoningEngine()
        analysis = engine.analyze_and_route('latest developments in AI applied to biology')
        results['reasoning'] = {'ok': True, 'detail': analysis}
    except Exception as e:
        logger.error("Reasoning layer test failed: %s", e)
        results['reasoning'] = {'ok': False, 'error': str(e)}

    # Action
    try:
        from workspace.bitcore_research import action_layer as action
        logger.info("Testing action layer (DRY_RUN=%s)", cfg.DRY_RUN)
        ae = action.ActionEngine()
        coll = ae.gather_data('machine learning applications in healthcare', 'scientific', max_results_per_source=2)
        results['action'] = {'ok': True, 'summary': coll.get('summary', {})}
    except Exception as e:
        logger.error("Action layer test failed: %s", e)
        results['action'] = {'ok': False, 'error': str(e)}

    # Reflection
    try:
        from workspace.bitcore_research import reflection_layer as reflection
        logger.info("Testing reflection layer")
        re_fl = reflection.ReflectionEngine()
        mock_research_data = {
            'query': 'test',
            'sources': {'core_api': {'count': 1, 'status': 'success'}},
            'summary': {'total_sources': 1, 'successful_sources': 1, 'total_results': 1}
        }
        ev = re_fl.evaluate_quality(mock_research_data, 'scientific')
        results['reflection'] = {'ok': True, 'overall_score': ev.get('overall_score')}
    except Exception as e:
        logger.error("Reflection layer test failed: %s", e)
        results['reflection'] = {'ok': False, 'error': str(e)}

    # Summarize
    success_count = sum(1 for v in results.values() if v.get('ok'))
    results['summary'] = {'success_count': success_count, 'component_count': 4}
    logger.info('Initialization summary: %s', results['summary'])
    return results


if __name__ == '__main__':
    run_initialization()
