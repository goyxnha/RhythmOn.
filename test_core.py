import pandas as pd
import pytest

from rhythmon_core import (
    FEATURES,
    TARGET,
    category_distribution,
    evaluate_models,
    normalize_dataset,
    predict_grade,
)


def sample_frame(n=40):
    return pd.DataFrame({
        "studytime": [1, 2, 3, 4] * (n // 4),
        "absences": list(range(n)),
        "failures": [0, 1, 2, 3] * (n // 4),
        "freetime": [1, 2, 3, 4, 5] * (n // 5) + [1] * (n % 5),
        "goout": [1, 2, 3, 4, 5] * (n // 5) + [1] * (n % 5),
        "G3": [float((i * 3) % 21) for i in range(n)],
    })


def test_normalize_selects_required_columns_and_preserves_valid_rows():
    source = sample_frame(40)
    source["school"] = "MS"
    report = normalize_dataset(source)
    assert len(report.data) == 40
    assert report.removed_rows == 0
    assert list(report.data.columns) == FEATURES + [TARGET]


def test_normalize_reports_invalid_rows_without_imputation():
    source = sample_frame(40)
    source.loc[0, "studytime"] = 8
    source.loc[1, "G3"] = None
    report = normalize_dataset(source)
    assert len(report.data) == 38
    assert report.removed_rows == 2


def test_missing_required_column_raises_error():
    source = sample_frame(40).drop(columns=["goout"])
    with pytest.raises(ValueError, match="필수 열"):
        normalize_dataset(source)


def test_category_distribution_sums_to_population():
    report = normalize_dataset(sample_frame(40))
    result = category_distribution(report.data, "studytime")
    assert int(result["학생 수"].sum()) == 40
    assert result["비율(%)"].sum() == pytest.approx(100.0)


def test_prediction_is_in_grade_range():
    report = normalize_dataset(sample_frame(40))
    values = {"studytime": 2, "absences": 3, "failures": 0, "freetime": 3, "goout": 2}
    prediction = predict_grade(report.data, "의사결정나무", values)
    assert 0 <= prediction <= 20


def test_all_models_return_cross_validation_metrics():
    report = normalize_dataset(sample_frame(60))
    result = evaluate_models(report.data)
    assert set(result.columns) == {"모델", "MAE", "RMSE", "R²"}
    assert len(result) == 3
    assert result["MAE"].ge(0).all()
    assert result["RMSE"].ge(0).all()
