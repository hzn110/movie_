# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)

MOVIES_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/kobis_movies.csv"
)

DAILY_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/kobis_daily.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():

    movies = pd.read_csv(
        MOVIES_URL,
        encoding="utf-8-sig"
    )

    daily = pd.read_csv(
        DAILY_URL,
        encoding="utf-8-sig"
    )

    return movies, daily


try:
    movies, daily = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 데이터 전처리
# =========================================================

movies.columns = movies.columns.str.strip()
daily.columns = daily.columns.str.strip()

# movieCd를 문자열로 통일
movies["movieCd"] = movies["movieCd"].astype(str).str.strip()
daily["영화코드"] = daily["영화코드"].astype(str).str.strip()

# 숫자형 변환
numeric_columns = [
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

for col in numeric_columns:
    movies[col] = pd.to_numeric(
        movies[col],
        errors="coerce"
    )

# 개봉일
movies["openDt"] = pd.to_datetime(
    movies["openDt"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

# 영화코드 정렬
movies = movies.sort_values(
    "movieCd",
    ascending=True
).reset_index(drop=True)


# =========================================================
# 기간 계산
# =========================================================

daily["날짜"] = pd.to_datetime(
    daily["날짜"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

period_start = daily["날짜"].min()
period_end = daily["날짜"].max()


# =========================================================
# 제목
# =========================================================

st.title("🎬 영화 흥행 예측기")

st.write(
    "KOBIS 영화 정보를 이용해 영화의 **총 관객 수(total_audi)**를 "
    "다중 회귀 모델로 예측합니다."
)


# =========================================================
# 데이터 기간 / 기본 정보
# =========================================================

st.subheader("📊 데이터 정보")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "영화 수",
    f"{len(movies):,}편"
)

c2.metric(
    "일별 데이터 기간",
    f"{period_start:%Y-%m-%d} ~ {period_end:%Y-%m-%d}"
)

c3.metric(
    "일별 데이터 행",
    f"{len(daily):,}"
)

c4.metric(
    "영화 정보 열 수",
    f"{len(movies.columns)}개"
)


# =========================================================
# 영화별 표의 맨 위 부분
# =========================================================

st.subheader("📋 영화별 데이터")

st.caption(
    "영화코드(movieCd) 오름차순으로 정렬한 영화별 데이터입니다."
)

# 원본 컬럼과 첫 데이터 행 표시
st.dataframe(
    movies.head(1),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 변수 선택
# =========================================================

st.subheader("⚙️ 예측 변수 선택")

st.write(
    "체크한 변수를 사용하여 다중 선형회귀 모델을 학습합니다. "
    "목표 변수는 `total_audi`(총 관객 수)입니다."
)


candidate_variables = [
    "openDt",
    "genre",
    "nation",
    "first_scrn",
    "first_show",
    "first_date",
    "peak",
    "first_week_audi",
    "days_in_top10"
]

variable_labels = {
    "openDt": "개봉일 (openDt)",
    "genre": "장르 (genre)",
    "nation": "국가 (nation)",
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "first_date": "10위권 첫 등장일",
    "peak": "성수기 개봉 여부",
    "first_week_audi": "첫 주 관객",
    "days_in_top10": "10위권 유지일"
}


# 날짜를 숫자로 변환한 임시 컬럼
movies_model = movies.copy()

movies_model["openDt_num"] = (
    movies_model["openDt"] -
    pd.Timestamp("2000-01-01")
).dt.days

movies_model["first_date_num"] = pd.to_datetime(
    movies_model["first_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

movies_model["first_date_num"] = (
    movies_model["first_date_num"] -
    pd.Timestamp("2000-01-01")
).dt.days


# 체크박스
selected_variables = []

cols = st.columns(3)

for i, variable in enumerate(candidate_variables):

    with cols[i % 3]:

        if st.checkbox(
            variable_labels[variable],
            value=True,
            key=f"variable_{variable}"
        ):
            selected_variables.append(variable)


if not selected_variables:
    st.warning("최소 한 개의 예측 변수를 선택해주세요.")
    st.stop()


# =========================================================
# 모델용 변수 변환
# =========================================================

feature_mapping = {
    "openDt": "openDt_num",
    "first_date": "first_date_num",
    "genre": "genre",
    "nation": "nation",
    "first_scrn": "first_scrn",
    "first_show": "first_show",
    "peak": "peak",
    "first_week_audi": "first_week_audi",
    "days_in_top10": "days_in_top10"
}

model_features = [
    feature_mapping[x]
    for x in selected_variables
]


# =========================================================
# 10편마다 앞 3편 테스트
# =========================================================

# 이미 movieCd 순으로 정렬되어 있음
movies_model = movies_model.reset_index(drop=True)

test_indices = []

for start in range(0, len(movies_model), 10):

    block_indices = list(
        range(
            start,
            min(start + 10, len(movies_model))
        )
    )

    # 각 10편 묶음에서 앞의 3편
    test_indices.extend(
        block_indices[:3]
    )


test_indices = sorted(set(test_indices))

test_mask = movies_model.index.isin(test_indices)

train_df = movies_model.loc[~test_mask].copy()
test_df = movies_model.loc[test_mask].copy()


# =========================================================
# 학습 데이터 / 테스트 데이터
# =========================================================

X_train = train_df[model_features]
y_train = train_df["total_audi"]

X_test = test_df[model_features]
y_test = test_df["total_audi"]


# =========================================================
# 결측치 및 범주형 변수 처리
# =========================================================

categorical_features = []

numeric_features = []

for feature in model_features:

    if movies_model[feature].dtype == "object":
        categorical_features.append(feature)

    else:
        numeric_features.append(feature)


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


transformers = []

if numeric_features:
    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    )

if categorical_features:
    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers
)


model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# =========================================================
# 모델 학습
# =========================================================

model.fit(
    X_train,
    y_train
)


# =========================================================
# 테스트 예측
# =========================================================

predictions = model.predict(X_test)

predictions = np.asarray(predictions)

# 음수 관객 수 방지
predictions = np.maximum(
    predictions,
    0
)


# =========================================================
# 평가
# =========================================================

r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


# =========================================================
# 결과 정보
# =========================================================

st.subheader("📈 학습 및 평가 결과")

r1, r2_col, r3, r4 = st.columns(4)

r1.metric(
    "전체 영화",
    f"{len(movies_model):,}편"
)

r2_col.metric(
    "학습에 사용한 영화",
    f"{len(train_df):,}편"
)

r3.metric(
    "평가한 영화",
    f"{len(test_df):,}편"
)

r4.metric(
    "R² 점수",
    f"{r2:.4f}"
)


st.info(
    f"**기준 기간:** {period_start:%Y-%m-%d} ~ {period_end:%Y-%m-%d}"
)


# =========================================================
# 평가 점수
# =========================================================

st.markdown("### 평가 점수")

m1, m2, m3 = st.columns(3)

m1.metric(
    "R²",
    f"{r2:.4f}"
)

m2.metric(
    "MAE",
    f"{mae:,.0f}명"
)

m3.metric(
    "RMSE",
    f"{rmse:,.0f}명"
)


st.caption(
    "R²는 높을수록 좋으며, MAE와 RMSE는 작을수록 좋습니다."
)


# =========================================================
# 실제값 / 예측값 / 오차
# =========================================================

result = test_df[
    [
        "movieCd",
        "movieNm",
        "total_audi"
    ]
].copy()

result["예측 총 관객 수"] = predictions

result["오차"] = (
    result["예측 총 관객 수"]
    - result["total_audi"]
)

result["절대 오차"] = (
    result["오차"]
    .abs()
)

result["오차율(%)"] = np.where(
    result["total_audi"] != 0,
    result["오차"]
    / result["total_audi"]
    * 100,
    np.nan
)

result = result.sort_values(
    "movieCd"
).reset_index(drop=True)


st.subheader("🎞️ 테스트 영화 예측 결과")

display_result = result.copy()

display_result["total_audi"] = (
    display_result["total_audi"]
    .round(0)
    .astype(int)
)

display_result["예측 총 관객 수"] = (
    display_result["예측 총 관객 수"]
    .round(0)
    .astype(int)
)

display_result["오차"] = (
    display_result["오차"]
    .round(0)
    .astype(int)
)

display_result["절대 오차"] = (
    display_result["절대 오차"]
    .round(0)
    .astype(int)
)

display_result["오차율(%)"] = (
    display_result["오차율(%)"]
    .round(2)
)

display_result = display_result.rename(
    columns={
        "movieCd": "영화코드",
        "movieNm": "영화명",
        "total_audi": "실제 총 관객 수",
        "오차율(%)": "오차율 (%)"
    }
)

st.dataframe(
    display_result,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 1,000명 미만 예측
# =========================================================

low_prediction_mask = (
    predictions < 1000
)

low_prediction_count = int(
    low_prediction_mask.sum()
)


# =========================================================
# Plotly 산점도
# =========================================================

st.subheader(
    "📊 실제 총 관객 수 vs 예측 총 관객 수"
)

st.write(
    "가로축은 실제 총 관객 수, "
    "세로축은 예측 총 관객 수입니다. "
    "두 축 모두 로그 스케일입니다."
)


fig = go.Figure()


# ---------------------------------------------------------
# 일반 테스트 영화
# ---------------------------------------------------------

normal_mask = predictions >= 1000

normal_result = result.loc[
    normal_mask
]

if len(normal_result) > 0:

    fig.add_trace(
        go.Scatter(
            x=normal_result["total_audi"],
            y=normal_result["예측 총 관객 수"],
            mode="markers",
            name="테스트 영화",
            text=normal_result["movieNm"],
            customdata=np.stack(
                [
                    normal_result["movieCd"],
                    normal_result["total_audi"],
                    normal_result["예측 총 관객 수"],
                    normal_result["오차"]
                ],
                axis=1
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제: %{customdata[1]:,.0f}명<br>"
                "예측: %{customdata[2]:,.0f}명<br>"
                "오차: %{customdata[3]:,.0f}명"
                "<extra></extra>"
            ),
            marker=dict(
                size=9
            )
        )
    )


# ---------------------------------------------------------
# 예측 1,000명 미만 영화
# ---------------------------------------------------------

low_result = result.loc[
    low_prediction_mask
].copy()

if len(low_result) > 0:

    # 로그 그래프에서 실제로 바닥에 붙여 보이도록
    # 1,000명 바로 아래의 고정 위치 사용
    floor_y = 900

    fig.add_trace(
        go.Scatter(
            x=low_result["total_audi"],
            y=[floor_y] * len(low_result),
            mode="markers",
            name="예측 1,000명 미만",
            text=low_result["movieNm"],
            customdata=np.stack(
                [
                    low_result["movieCd"],
                    low_result["total_audi"],
                    low_result["예측 총 관객 수"],
                    low_result["오차"]
                ],
                axis=1
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제: %{customdata[1]:,.0f}명<br>"
                "예측: %{customdata[2]:,.0f}명<br>"
                "오차: %{customdata[3]:,.0f}명"
                "<extra></extra>"
            ),
            marker=dict(
                symbol="triangle-down",
                size=11
            )
        )
    )


# ---------------------------------------------------------
# 대각선 y=x
# ---------------------------------------------------------

positive_actual = result.loc[
    result["total_audi"] > 0,
    "total_audi"
]

positive_pred = result.loc[
    result["예측 총 관객 수"] > 0,
    "예측 총 관객 수"
]

if len(positive_actual) > 0:

    min_value = min(
        positive_actual.min(),
        positive_pred.min()
    )

    max_value = max(
        positive_actual.max(),
        positive_pred.max()
    )

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="실제값 = 예측값",
            line=dict(
                dash="dash"
            )
        )
    )


fig.update_layout(
    xaxis=dict(
        title="실제 총 관객 수",
        type="log"
    ),
    yaxis=dict(
        title="예측 총 관객 수",
        type="log"
    ),
    height=650,
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


st.info(
    f"예측 총 관객 수가 1,000명보다 작게 나온 영화: "
    f"**{low_prediction_count}편**"
)


# =========================================================
# 선택된 변수
# =========================================================

with st.expander("🔎 현재 모델에 사용된 변수"):

    for variable in selected_variables:
        st.write(
            f"- {variable_labels[variable]}"
        )


# =========================================================
# 전체 영화 데이터
# =========================================================

with st.expander("📚 전체 영화 데이터 보기"):

    st.dataframe(
        movies,
        use_container_width=True,
        hide_index=True
    )
