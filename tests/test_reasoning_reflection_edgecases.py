from workspace.bitcore_research.reasoning_layer import ReasoningEngine
from workspace.bitcore_research.reflection_layer import ReflectionEngine
from workspace.config import cfg


def test_reasoning_complexity_boundaries(tmp_path):
    cfg.DRY_RUN = True
    cfg.KNOWLEDGE_BASE_PATH = str(tmp_path / "kb.json")
    cfg.ASSURANCE_REPORTS_DIR = str(tmp_path / "assurance")
    cfg.TMP_DIR = str(tmp_path / "tmp")

    engine = ReasoningEngine()

    # Simple: very short, non-technical
    simple = "AI benefits"
    res = engine.analyze_and_route(simple)
    assert res['complexity'] == 'simple'

    # Moderate: mentions two domains -> interdisciplinary
    moderate = "Interplay between computer science and biology for genome analysis using neural networks"
    res = engine.analyze_and_route(moderate)
    assert res['complexity'] in ('moderate', 'complex')

    # Complex: long, technical, multi-domain
    complex_q = (
        "Design of quantum cryptographic protocols integrating neural transformer architectures "
        "for secure distributed machine learning across medical imaging datasets and population genomics"
    )
    res = engine.analyze_and_route(complex_q)
    assert res['complexity'] in ('complex', 'expert')


def test_reflection_high_scores_produce_excellent(tmp_path):
    cfg.DRY_RUN = True
    cfg.KNOWLEDGE_BASE_PATH = str(tmp_path / "kb.json")
    cfg.ASSURANCE_REPORTS_DIR = str(tmp_path / "assurance")
    cfg.TMP_DIR = str(tmp_path / "tmp")

    refl = ReflectionEngine()

    high_quality_data = {
        'query': 'test high quality',
        'sources': {
            'core_api': {'count': 5},
            'arxiv': {'count': 5},
            'pubmed': {'count': 5},
            'google_scholar': {'count': 5}
        },
        'summary': {'total_sources': 4, 'successful_sources': 4, 'total_results': 20}
    }

    eval_res = refl.evaluate_quality(high_quality_data, 'scientific')
    assert 'overall_score' in eval_res
    assert eval_res['overall_score'] >= 0.8
    assert eval_res['verdict'] == 'excellent'
