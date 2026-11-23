# 🎵 RhythmOn | 학습 습관 분석 AI

학생의 학습 습관과 생활 패턴을 분석해 최종 수학 성적(G3)을 추정하는 교육용 AI 웹 프로토타입입니다.

> 이 앱은 학교 연구·시연 목적으로 제작되었습니다. 예측값은 참고용이며 실제 성적을 보장하지 않습니다. 학습 습관과 성적 사이의 인과관계를 입증하지 않습니다.

## 화면 미리보기

![RhythmOn 화면 미리보기](assets/rhythmon_preview.png)

실행 후 실제 데이터가 연결되면 학생 수, 범주별 비율, 결석 분포, 모델 성능 비교 그래프가 원자료를 기준으로 계산됩니다. 위 이미지는 프로그램의 화면 구성을 보여주는 디자인 미리보기입니다.

## 핵심 기능

- UCI Student Performance 수학 자료(`student-mat.csv`) 자동 다운로드 또는 CSV/Excel 직접 업로드
- 5개 입력 속성의 실제 학생 수·비율 분포 시각화
- 결석 횟수 분포 히스토그램
- 의사결정나무 회귀, 선형회귀, 서포트 벡터 회귀(SVR) 비교
- 동일한 5겹 교차검증으로 MAE, RMSE, R² 평가
- 학생 입력값에 따른 최종 수학 성적(G3) 추정
- 학습 습관 점검 제안 및 결과 CSV 저장

## 사용 데이터와 변수

출처: UCI Machine Learning Repository, [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)

원 연구: Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance*. Proceedings of the 5th Annual Future Business Technology Conference.

| 원자료 변수 | 속성 | 앱에서의 역할 |
|---|---|---|
| `studytime` | 주간 학습 시간(1~4) | 입력 변수 |
| `absences` | 결석 횟수 | 입력 변수 |
| `failures` | 과거 낙제 횟수 범주(0~4) | 입력 변수 |
| `freetime` | 방과 후 여가 시간(1~5) | 입력 변수 |
| `goout` | 친구와 외출 빈도(1~5) | 입력 변수 |
| `G3` | 최종 수학 성적(0~20점) | 예측 대상 |

`G1`, `G2`는 학습 습관 중심의 예측 실험을 위해 입력 변수에서 제외했습니다.

## 로컬 실행 방법

Python 3.11 또는 3.12를 권장합니다. Streamlit Community Cloud에 배포할 때는 배포 화면의 **Advanced settings**에서 Python 3.12를 선택하세요.

### Windows

1. 이 저장소를 내려받거나 `Code → Download ZIP`으로 압축 파일을 받습니다.
2. 압축을 푼 폴더에서 터미널(명령 프롬프트)을 엽니다.
3. 아래 명령을 순서대로 실행합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 Streamlit 로컬 주소가 열리면 왼쪽 메뉴에서 UCI 공개 수학 자료를 불러오면 됩니다. 네트워크 연결이 제한되면 UCI에서 `student-mat.csv`를 내려받아 직접 업로드하세요.

## GitHub에 저장하기

1. GitHub에서 새 저장소를 만듭니다. 예: `rhythmon-learning-ai`.
2. 저장소 공개 여부를 선택합니다. 공개 저장소라면 누구나 소스 코드를 볼 수 있으므로 개인 정보나 비밀키를 올리지 마세요.
3. 이 프로젝트 폴더 안의 파일과 폴더를 저장소 최상위에 업로드합니다.
4. `app.py`, `rhythmon_core.py`, `requirements.txt`, `README.md`가 저장소 첫 화면에서 보이는지 확인합니다.

Git 명령을 사용할 수 있다면 프로젝트 폴더에서 다음 명령을 실행할 수도 있습니다.

```bash
git init
git add .
git commit -m "Initial RhythmOn learning analytics app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/rhythmon-learning-ai.git
git push -u origin main
```

`YOUR-USERNAME`은 본인의 GitHub 아이디로 바꾸세요.

## 웹에서 공개하기: Streamlit Community Cloud

1. GitHub에 프로젝트를 올립니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에 접속해 GitHub 계정으로 로그인합니다.
3. **Create app**을 선택합니다.
4. 저장소, 브랜치(`main`), 앱 파일 경로(`app.py`)를 지정합니다.
5. **Advanced settings**에서 Python 3.12를 선택한 뒤 **Deploy**를 누릅니다.

공식 안내: [Deploy your app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## 프로젝트 구조

```text
RhythmOn_GitHub/
├── app.py                   # Streamlit 웹 화면
├── rhythmon_core.py         # 데이터 전처리와 AI 모델 기능
├── requirements.txt         # 파이썬 라이브러리
├── .gitignore               # Git 제외 파일
├── .streamlit/config.toml   # 화면 테마 및 서버 설정
├── assets/                  # 화면 미리보기 이미지
├── data/                    # 데이터 사용 안내
└── tests/                   # 핵심 기능 테스트
```

## 데이터 처리와 윤리

- 필수 열 누락, 결측값, 허용 범위를 벗어난 행은 모델 학습 전에 제외하고 제외 행 수를 표시합니다. 임의로 데이터를 생성하거나 평균값으로 채우지 않습니다.
- 모델 평가는 같은 입력 변수와 같은 5겹 교차검증 조건으로 비교합니다.
- 원자료는 포르투갈 중등학교 학생의 자료입니다. 한국 학생에게 그대로 일반화할 수 없으며, 한국 학생 자료로 외부 검증하기 전에는 실제 성적 예측 서비스로 사용하면 안 됩니다.
- 이름, 주소, 연락처 등 개인 식별 정보를 수집하지 않습니다. 입력값은 앱에서 서버·DB에 자동 저장하지 않습니다.
- 모델의 예측은 참고용 추정치로만 제공하며 학생의 능력이나 미래를 단정하지 않습니다.

## 라이선스

이 프로젝트 소스 코드는 MIT License로 배포합니다. 데이터셋은 UCI 원 출처 및 해당 저장소의 이용 조건을 확인해 사용하세요.
