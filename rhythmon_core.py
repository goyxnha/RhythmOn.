"""Data and machine-learning utilities for RhythmOn."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

FEATURES = ["studytime", "absences", "failures", "freetime", "goout"]
TARGET = "G3"
REQUIRED_COLUMNS = FEATURES + [TARGET]

KOREAN_NAMES = {
    "studytime": "주간 학습 시간",
    "absences": "결석 횟수",
    "failures": "과거 낙제 횟수",
    "freetime": "방과 후 여가 시간",
    "goout": "친구와 외출 빈도",
    "G3": "최종 수학 성적",
}

STUDYTIME_LABELS = {
    1: "2시간 미만",
    2: "2~5시간",
    3: "5~10시간",
    4: "10시간 초과",
}
LEVEL_LABELS = {
    1: "매우 낮음",
    2: "낮음",
    3: "보통",
    4: "높음",
    5: "매우 높음",
}
FAILURE_LABELS = {0: "0회", 1: "1회", 2: "2회", 3: "3회", 4: "4회 이상"}
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student.zip"
UCI_PAGE_URL = "https://archive.ics.uci.edu/dataset/320/student+performance"


@dataclass
class DatasetReport:
    data: pd.DataFrame
    original_rows: int
    removed_rows: int


def normalize_dataset(raw: pd.DataFrame) -> DatasetReport:
    """Select required columns, convert types, and remove invalid/incomplete rows.

    No missing or invalid values are imputed. Rows that cannot be used are excluded
    and reported, so the app never fabricates training examples.
    """
    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "필수 열이 없습니다: " + ", ".join(missing_columns)
            + ". UCI student-mat.csv 파일인지 확인해 주세요."
        )

    original_rows = len(frame)
    frame = frame[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    integer_columns = ["studytime", "absences", "failures", "freetime", "goout"]
    integer_mask = pd.Series(True, index=frame.index)
    for column in integer_columns:
        integer_mask &= frame[column].notna() & np.isclose(
            frame[column].fillna(-999), np.round(frame[column].fillna(-999))
        )

    valid_mask = (
        integer_mask
        & frame["studytime"].isin([1, 2, 3, 4])
        & frame["absences"].ge(0)
        & frame["failures"].between(0, 4)
        & frame["freetime"].between(1, 5)
        & frame["goout"].between(1, 5)
        & frame[TARGET].between(0, 20)
    )
    clean = frame.loc[valid_mask].copy().reset_index(drop=True)
    for column in integer_columns:
        clean[column] = clean[column].astype(int)

    if len(clean) < 25:
        raise ValueError(
            f"사용 가능한 행이 {len(clean)}개뿐입니다. 모델 학습을 위해 최소 25개 이상의 유효한 행이 필요합니다."
        )
    return DatasetReport(
        data=clean,
        original_rows=original_rows,
        removed_rows=original_rows - len(clean),
    )


def load_uci_math_dataset(timeout: int = 30) -> DatasetReport:
    """Download the official UCI ZIP and read student-mat.csv only."""
    response = requests.get(UCI_ZIP_URL, timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        matches = [
            name for name in archive.namelist()
            if name.lower().endswith("student-mat.csv")
        ]
        if not matches:
            raise ValueError("UCI 압축 파일에서 student-mat.csv를 찾지 못했습니다.")
        with archive.open(matches[0]) as csv_file:
            raw = pd.read_csv(csv_file, sep=";")
    return normalize_dataset(raw)


def parse_uploaded_dataset(file_bytes: bytes, filename: str) -> DatasetReport:
    """Read semicolon/comma CSV or Excel upload and validate it."""
    lower_name = filename.lower()
    if lower_name.endswith(".xlsx"):
        raw = pd.read_excel(io.BytesIO(file_bytes))
    elif lower_name.endswith(".csv"):
        try:
            raw = pd.read_csv(io.BytesIO(file_bytes), sep=";")
            if len(raw.columns) == 1:
                raw = pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError:
            raw = pd.read_csv(io.BytesIO(file_bytes), sep=";", encoding="latin-1")
            if len(raw.columns) == 1:
                raw = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
    else:
        raise ValueError("CSV 또는 Excel 파일만 업로드할 수 있습니다.")
    return normalize_dataset(raw)


def build_models() -> dict[str, Any]:
    """Return comparable regression models."""
    return {
        "의사결정나무": DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=8, random_state=42
        ),
        "선형회귀": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]),
        "서포트 벡터 회귀(SVR)": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", SVR(C=10, epsilon=1.0, kernel="rbf")),
        ]),
    }


def evaluate_models(data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all models with identical shuffled 5-fold cross-validation."""
    if len(data) < 25:
        raise ValueError("5겹 교차검증을 위해 최소 25개 이상의 데이터가 필요합니다.")
    x = data[FEATURES]
    y = data[TARGET]
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    rows: list[dict[str, float | str]] = []

    for model_name, model in build_models().items():
        scores = cross_validate(
            model,
            x,
            y,
            cv=folds,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=1,
            error_score="raise",
        )
        rows.append({
            "모델": model_name,
            "MAE": float(-scores["test_mae"].mean()),
            "RMSE": float(-scores["test_rmse"].mean()),
            "R²": float(scores["test_r2"].mean()),
        })
    return pd.DataFrame(rows).sort_values("MAE", ascending=True).reset_index(drop=True)


def predict_grade(
    data: pd.DataFrame, model_name: str, input_values: dict[str, int]
) -> float:
    """Fit the selected model on the available dataset and predict G3."""
    models = build_models()
    if model_name not in models:
        raise ValueError(f"알 수 없는 모델입니다: {model_name}")
    input_frame = pd.DataFrame([input_values], columns=FEATURES)
    model = models[model_name]
    model.fit(data[FEATURES], data[TARGET])
    prediction = float(model.predict(input_frame)[0])
    return float(np.clip(prediction, 0.0, 20.0))


def category_distribution(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Return category counts and percentages for ordinal/categorical features."""
    label_maps: dict[str, dict[int, str]] = {
        "studytime": STUDYTIME_LABELS,
        "failures": FAILURE_LABELS,
        "freetime": LEVEL_LABELS,
        "goout": LEVEL_LABELS,
    }
    if feature not in label_maps:
        raise ValueError("범주형 분포 그래프에 지원되지 않는 속성입니다.")
    counts = data[feature].value_counts().sort_index()
    labels = label_maps[feature]
    result = pd.DataFrame({
        "범주": [labels.get(int(value), str(value)) for value in counts.index],
        "학생 수": counts.values.astype(int),
    })
    result["비율(%)"] = result["학생 수"] / len(data) * 100
    return result


def make_recommendations(input_values: dict[str, int], data: pd.DataFrame) -> list[str]:
    """Create non-diagnostic study-planning prompts based on entered habits."""
    tips: list[str] = []
    median_absences = float(data["absences"].median())

    if input_values["studytime"] <= 2:
        tips.append(
            "학습 시간을 일정하게 확보할 수 있도록 짧고 구체적인 학습 목표부터 세워 보세요."
        )
    else:
        tips.append(
            "현재 입력한 학습 시간대를 유지하면서 복습과 오답 점검이 함께 이루어지는지 살펴보세요."
        )

    if input_values["absences"] > median_absences:
        tips.append(
            f"입력한 결석 횟수는 참고 데이터의 중앙값({median_absences:.0f}회)보다 많습니다. 빠진 수업의 학습 공백을 확인해 보세요."
        )
    else:
        tips.append(
            "입력한 결석 횟수는 참고 데이터의 중앙값 이하입니다. 수업 내용을 꾸준히 정리하는 습관을 이어가 보세요."
        )

    if input_values["failures"] > 0:
        tips.append(
            "이전에 어려움을 겪었던 과목이나 단원을 작은 목표로 나누어 다시 점검해 보세요."
        )
    if input_values["freetime"] >= 4:
        tips.append(
            "여가 시간을 없애기보다 학습 시간과 휴식 시간을 구분해 균형 있게 계획해 보세요."
        )
    if input_values["goout"] >= 4:
        tips.append(
            "친구와의 활동을 유지하면서 과제와 복습 일정을 미리 기록해 보세요."
        )
    return tips[:5]
