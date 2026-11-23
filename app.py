from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rhythmon_core import (
    FEATURES,
    KOREAN_NAMES,
    STUDYTIME_LABELS,
    LEVEL_LABELS,
    FAILURE_LABELS,
    build_models,
    category_distribution,
    evaluate_models,
    load_uci_math_dataset,
    make_recommendations,
    parse_uploaded_dataset,
    predict_grade,
)

st.set_page_config(
    page_title="RhythmOn | 학습 습관 분석 AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(show_spinner=False)
def cached_model_evaluation(data: pd.DataFrame) -> pd.DataFrame:
    return evaluate_models(data)

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #10213f; }
    [data-testid="stSidebar"] * { color: #eef4ff !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section { background: #1b3156; }
    .rhythm-brand { color:#244b84; font-size:2.15rem; font-weight:850; letter-spacing:-1.2px; }
    .rhythm-sub { color:#64748b; font-size:1rem; margin-top:-9px; }
    .eyebrow { color:#4b74ae; font-size:.78rem; font-weight:750; letter-spacing:.09em; }
    .panel { background:white; border:1px solid #e6ebf2; border-radius:18px; padding:18px 20px; }
    .muted { color:#64748b; }
    div[data-testid="stMetric"] { background:white; border:1px solid #e6ebf2; padding:15px 17px; border-radius:15px; }
    div.stButton > button[kind="primary"] { background:#315f9d; border-color:#315f9d; }
    .stTabs [data-baseweb="tab-list"] { gap:8px; }
    .stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:10px 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def set_dataset(report, origin: str) -> None:
    st.session_state.rhythm_data = report.data
    st.session_state.rhythm_origin = origin
    st.session_state.rhythm_removed = report.removed_rows
    st.session_state.rhythm_original_rows = report.original_rows


if "rhythm_data" not in st.session_state:
    st.session_state.rhythm_data = None
    st.session_state.rhythm_origin = None
    st.session_state.rhythm_removed = 0
    st.session_state.rhythm_original_rows = 0

with st.sidebar:
    st.markdown("# 🎵 RhythmOn")
    st.caption("학습 습관 분석 · 성적 예측")
    st.markdown("---")
    st.markdown("### 데이터 연결")
    uploaded_file = st.file_uploader(
        "원자료 파일 업로드",
        type=["csv", "xlsx"],
        help="UCI student-mat.csv 또는 필요한 열을 포함한 CSV/Excel 파일을 선택하세요.",
    )
    if uploaded_file is not None:
        try:
            report = parse_uploaded_dataset(uploaded_file.getvalue(), uploaded_file.name)
            set_dataset(report, f"업로드 파일: {uploaded_file.name}")
            st.success(f"{len(report.data)}개 행을 불러왔습니다.")
        except Exception as exc:
            st.error(str(exc))

    if st.button("UCI 공개 수학 자료 불러오기", use_container_width=True):
        try:
            with st.spinner("UCI 원자료를 불러오는 중입니다..."):
                report = load_uci_math_dataset()
            set_dataset(report, "UCI Student Performance · Mathematics")
            st.success(f"{len(report.data)}명 자료 연결 완료")
        except Exception as exc:
            st.error(f"자동 연결에 실패했습니다: {exc}")

    st.markdown("---")
    st.markdown("### AI 입력 변수")
    for item in ["주간 학습 시간", "결석 횟수", "과거 낙제 횟수", "방과 후 여가 시간", "친구와 외출 빈도"]:
        st.caption(f"• {item}")
    st.caption("예측 대상: 최종 수학 성적 G3")

st.markdown('<div class="eyebrow">PERSONAL LEARNING ANALYTICS</div>', unsafe_allow_html=True)
st.markdown('<div class="rhythm-brand">RhythmOn</div>', unsafe_allow_html=True)
st.markdown('<div class="rhythm-sub">나의 학습 리듬을 데이터로 살펴보고, 다음 학습 계획의 힌트를 찾아보세요.</div>', unsafe_allow_html=True)
st.write("")

if st.session_state.rhythm_data is None:
    st.info("분석을 시작하려면 왼쪽 메뉴에서 UCI 공개 자료를 불러오거나 student-mat.csv 파일을 업로드하세요.")
    st.markdown("### RhythmOn 이용 흐름")
    stages = st.columns(4)
    stage_data = [
        ("01", "데이터 수집", "실제 학생 학습 습관 자료 연결"),
        ("02", "데이터 분석", "5개 입력 속성의 분포 시각화"),
        ("03", "AI 모델 비교", "회귀 모델 3종의 성능 검증"),
        ("04", "성적 예측", "학습 습관 입력 후 G3 추정"),
    ]
    for col, (num, title, desc) in zip(stages, stage_data):
        with col:
            st.markdown(
                f'<div class="panel"><div class="eyebrow">STEP {num}</div><h3>{title}</h3><p class="muted">{desc}</p></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    st.markdown("### 분석에 사용하는 속성")
    info_cols = st.columns(5)
    feature_notes = [
        ("📘", "주간 학습 시간", "1~4단계"),
        ("🗓️", "결석 횟수", "실제 횟수"),
        ("📚", "과거 낙제 횟수", "0~4 범주"),
        ("🌿", "방과 후 여가", "1~5단계"),
        ("👥", "친구와 외출", "1~5단계"),
    ]
    for col, (icon, title, desc) in zip(info_cols, feature_notes):
        with col:
            st.markdown(
                f'<div class="panel"><h2>{icon}</h2><b>{title}</b><p class="muted">{desc}</p></div>',
                unsafe_allow_html=True,
            )
    st.caption("자료: Cortez & Silva (2008), UCI Student Performance Dataset. 분석 자료는 포르투갈 학생을 대상으로 수집되었으며 한국 학생에게 검증된 모델은 아닙니다.")
    st.stop()

# Data is connected
student_data: pd.DataFrame = st.session_state.rhythm_data.copy()
source_name = st.session_state.rhythm_origin or "연결된 자료"

metric_cols = st.columns(4)
metric_cols[0].metric("분석 학생 수", f"{len(student_data):,}명")
metric_cols[1].metric("입력 속성", "5개")
metric_cols[2].metric("예측 대상", "G3")
metric_cols[3].metric("제외된 행", f"{st.session_state.rhythm_removed:,}개")
st.caption(f"데이터 출처: {source_name}  |  가공 전 {st.session_state.rhythm_original_rows:,}행 · 유효 데이터 {len(student_data):,}행")

TAB_DASH, TAB_MODELS, TAB_PREDICT, TAB_RESEARCH = st.tabs(
    ["📊 데이터 분석", "🤖 AI 모델 비교", "🎯 성적 예측", "📘 연구 정보"]
)

with TAB_DASH:
    st.subheader("학습 습관 데이터 분포")
    st.write("각 범주에 속한 학생 수와 전체 비율을 확인합니다. 그래프는 연결된 원자료에서 매번 계산합니다.")
    label_map = {
        "studytime": STUDYTIME_LABELS,
        "failures": FAILURE_LABELS,
        "freetime": LEVEL_LABELS,
        "goout": LEVEL_LABELS,
    }
    distribution_features = ["studytime", "failures", "freetime", "goout"]
    for start in range(0, len(distribution_features), 2):
        chart_cols = st.columns(2)
        for col, feature in zip(chart_cols, distribution_features[start:start + 2]):
            distribution = category_distribution(student_data, feature)
            fig = px.bar(
                distribution,
                x="범주",
                y="학생 수",
                text=distribution.apply(
                    lambda row: f"{int(row['학생 수'])}명 · {row['비율(%)']:.1f}%", axis=1
                ),
                title=KOREAN_NAMES[feature],
                color="학생 수",
                color_continuous_scale="Blues",
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                height=355,
                margin=dict(l=8, r=8, t=58, b=8),
                xaxis_title="응답 범주",
                yaxis_title="학생 수",
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            with col:
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(distribution, hide_index=True, use_container_width=True)

    st.subheader("결석 횟수 분포")
    absence_fig = px.histogram(
        student_data,
        x="absences",
        nbins=20,
        labels={"absences": "결석 횟수", "count": "학생 수"},
        title="결석 횟수별 학생 분포",
        color_discrete_sequence=["#4778b8"],
    )
    absence_fig.update_layout(height=330, bargap=0.06, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(absence_fig, use_container_width=True)

    with st.expander("가공 데이터 일부 확인"):
        view = student_data.rename(columns=KOREAN_NAMES)
        st.dataframe(view.head(25), hide_index=True, use_container_width=True)
    st.download_button(
        "가공 데이터 CSV 다운로드",
        student_data.to_csv(index=False).encode("utf-8-sig"),
        file_name="rhythmon_processed_data.csv",
        mime="text/csv",
    )

with TAB_MODELS:
    st.subheader("AI 회귀 모델 성능 비교")
    st.write("동일한 5개 입력 속성과 동일한 5겹 교차검증으로 세 모델을 비교합니다.")
    with st.spinner("교차검증을 실행하고 있습니다..."):
        model_results = cached_model_evaluation(student_data)
    best_model = str(model_results.iloc[0]["모델"])
    a, b, c = st.columns(3)
    for metric_col, metric_name, help_text in [
        (a, "MAE", "평균 절대 오차. 작을수록 좋습니다."),
        (b, "RMSE", "제곱근 평균 제곱 오차. 작을수록 좋습니다."),
        (c, "R²", "결과의 설명력을 나타내며 높을수록 좋습니다."),
    ]:
        row = model_results.loc[model_results["모델"] == best_model].iloc[0]
        metric_col.metric(metric_name, f"{row[metric_name]:.3f}", help=help_text)
    st.success(f"현재 교차검증에서 MAE가 가장 낮은 모델: {best_model}")
    st.dataframe(
        model_results.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    error_long = model_results.melt(
        id_vars="모델", value_vars=["MAE", "RMSE"], var_name="평가지표", value_name="오차"
    )
    error_fig = px.bar(
        error_long,
        x="모델",
        y="오차",
        color="평가지표",
        barmode="group",
        title="모델별 예측 오차 비교",
        color_discrete_sequence=["#4778b8", "#93b7df"],
    )
    error_fig.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(error_fig, use_container_width=True)
    st.info("교차검증 결과는 포르투갈 학생 자료에서의 성능입니다. 한국 학생에게 더 정확하다는 의미가 아니며, 별도의 한국 학생 자료로 외부 검증이 필요합니다.")

with TAB_PREDICT:
    st.subheader("나의 학습 습관 입력")
    st.write("다섯 가지 입력값을 선택하고 AI 모델을 지정하면 최종 수학 성적 G3를 추정합니다.")
    model_names = list(build_models().keys())
    default_model = str(model_results.iloc[0]["모델"])
    selected_model = st.selectbox(
        "예측 모델",
        model_names,
        index=model_names.index(default_model),
    )
    with st.form("rhythmon_prediction"):
        left, right = st.columns(2)
        with left:
            studytime = st.select_slider(
                "주간 학습 시간",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda value: f"{value}단계 · {STUDYTIME_LABELS[value]}",
            )
            absences = st.number_input(
                "결석 횟수",
                min_value=0,
                max_value=200,
                value=int(student_data["absences"].median()),
                step=1,
            )
            failures = st.selectbox(
                "과거 낙제 횟수",
                options=[0, 1, 2, 3, 4],
                format_func=lambda value: FAILURE_LABELS[value],
            )
        with right:
            freetime = st.select_slider(
                "방과 후 여가 시간",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda value: f"{value}단계 · {LEVEL_LABELS[value]}",
            )
            goout = st.select_slider(
                "친구와 외출 빈도",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda value: f"{value}단계 · {LEVEL_LABELS[value]}",
            )
        submitted = st.form_submit_button("AI 성적 예측하기", type="primary", use_container_width=True)

    if submitted:
        input_values = {
            "studytime": int(studytime),
            "absences": int(absences),
            "failures": int(failures),
            "freetime": int(freetime),
            "goout": int(goout),
        }
        prediction = predict_grade(student_data, selected_model, input_values)
        first, second = st.columns([1, 1.3])
        with first:
            st.metric("AI 예측 최종 수학 성적", f"{prediction:.1f}점")
            st.caption(f"사용 모델: {selected_model}")
        with second:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={"suffix": "점", "font": {"size": 34}},
                gauge={
                    "axis": {"range": [0, 20]},
                    "bar": {"color": "#315f9d"},
                    "steps": [
                        {"range": [0, 7], "color": "#fce8e6"},
                        {"range": [7, 14], "color": "#fff4d6"},
                        {"range": [14, 20], "color": "#e4f2e8"},
                    ],
                },
                title={"text": "예측 점수 · 0~20점"},
            ))
            gauge.update_layout(height=255, margin=dict(l=18, r=18, t=55, b=8), paper_bgcolor="white")
            st.plotly_chart(gauge, use_container_width=True)
        st.subheader("학습 습관 점검 제안")
        for recommendation in make_recommendations(input_values, student_data):
            st.markdown(f"- {recommendation}")
        prediction_export = pd.DataFrame([input_values])
        prediction_export["모델"] = selected_model
        prediction_export["예측_G3"] = prediction
        st.download_button(
            "입력값과 예측 결과 CSV 저장",
            prediction_export.to_csv(index=False).encode("utf-8-sig"),
            file_name="rhythmon_prediction.csv",
            mime="text/csv",
        )
        st.warning("예측값은 포르투갈 학생 자료로 학습한 모델의 참고용 추정치입니다. 실제 성적을 보장하거나 학생의 능력을 평가하지 않습니다. 생활 습관이 성적의 원인이라고 단정할 수도 없습니다.")

with TAB_RESEARCH:
    st.subheader("연구 설계와 데이터 윤리")
    st.markdown("**연구 목적**  \n    학생의 학습 습관과 생활 패턴을 분석하여 최종 수학 성적을 추정하고, 자신의 학습 패턴을 점검할 참고 정보를 제공합니다.")
    st.markdown("**입력 변수 및 예측 대상**")
    variable_table = pd.DataFrame([
        ["studytime", "주간 학습 시간", "1~4단계", "입력 변수"],
        ["absences", "결석 횟수", "0 이상의 정수", "입력 변수"],
        ["failures", "과거 낙제 횟수", "0~4 범주", "입력 변수"],
        ["freetime", "방과 후 여가 시간", "1~5단계", "입력 변수"],
        ["goout", "친구와 외출 빈도", "1~5단계", "입력 변수"],
        ["G3", "최종 수학 성적", "0~20점", "예측 정답"],
    ], columns=["원자료 변수", "한국어 속성", "값의 범위", "역할"])
    st.dataframe(variable_table, hide_index=True, use_container_width=True)
    st.markdown("**데이터 처리 원칙**")
    st.markdown(
        "- 결측값이나 허용 범위를 벗어난 값은 임의로 채우지 않고 제외합니다.\n"
        "- 이름, 연락처, 주소 등 개인 식별 정보는 요구하지 않습니다.\n"
        "- 입력한 예측값은 현재 브라우저 세션에서 처리하며 파일로 자동 전송하거나 저장하지 않습니다.\n"
        "- 학습 습관과 성적의 연관성을 살펴보는 도구이며 인과관계를 입증하지 않습니다."
    )
    st.markdown("**자료 출처**")
    st.markdown(
        "Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance*. "
        "Proceedings of the 5th Annual Future Business Technology Conference.\n\n"
        "[UCI Machine Learning Repository – Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)"
    )

st.markdown("---")
st.caption("RhythmOn · 학습 습관 분석 AI | 학교 연구 및 시연용 프로토타입")
