from workspace.bitcore_research.reflection import (
    QualityDimension,
    calculate_overall_score,
    determine_verdict,
    evaluate_quality_scores,
    generate_recommendations,
)


def test_evaluate_quality_scores_zero_sources():
    research_data = {"sources": {}}

    scores = evaluate_quality_scores(research_data)

    assert all(dimension in scores for dimension in QualityDimension)
    assert all(score == 0.0 for score in scores.values())

    overall = calculate_overall_score(scores)
    assert overall == 0.0
    assert determine_verdict(overall) == "needs_improvement"

    recommendations = generate_recommendations(scores, overall)
    assert "Increase the number of sources and total collected results" in recommendations
