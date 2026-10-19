import math

from features import compute_features

BASE = {
    "code_module": "AAA", "weekly_clicks": [10, 0, 5, 0], "weekly_active_days": [2, 0, 1, 0],
    "assessments_due": 2, "assessment_scores": [70.0], "late_submissions": 1,
}


def test_activity_features():
    f = compute_features(BASE)
    assert f["active_weeks"] == 2
    assert f["active_days"] == 3
    assert f["weeks_since_last_activity"] == 1
    assert math.isclose(f["clicks_total_log"], math.log1p(15))


def test_assessment_features():
    f = compute_features(BASE)
    assert f["assessments_submitted"] == 1
    assert f["assessments_missed"] == 1
    assert f["score_mean"] == 70.0
    assert f["late_submissions"] == 1


def test_missing_values_are_handled():
    f = compute_features({"code_module": "AAA", "weekly_clicks": [0, 0], "weekly_active_days": [0, 0]})
    assert f["gender"] == "unknown"
    assert f["imd_band"] == "unknown"
    assert f["score_mean"] == 0.0
    assert f["weeks_since_last_activity"] == 2
