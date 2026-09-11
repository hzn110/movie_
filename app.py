import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# 데이터 주소
# =========================================================

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
    movies_raw, daily_raw = load_data()

except Exception as e:

    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 원본 데이터 보관
# =========================================================

movies_original = movies_raw.copy()
daily = daily_raw.copy()


# =========================================================
# 열 이름 정리
# =========================================================

movies = movies_raw.copy()

movies.columns = movies.columns.str.strip()
daily.columns = daily.columns.str.strip()


# =========================================================
# 영화 정보 표의 첫 행을 위한 원본 데이터
# =========================================================

original_top_row = movies.head(1).copy()


# =========================================================
# movieCd 통일
# =========================================================

movies["movieCd"] = (
    movies["movieCd"]
    .astype(str)
    .str.strip()
)

daily["영화코드"] = (
    daily["영화코드"]
    .astype(str)
    .str.strip()
)


# =========================================================
# 일별 데이터 날짜
# =========================================================

daily["날짜"] = pd.to_datetime(
    daily["날짜"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

period_start = daily["날짜"].min()
period_end = daily["날짜"].max()


# =========================================================
# 영화 코드 순 정렬
# =========================================================

movies = (
    movies
    .sort_values("movieCd")
    .reset_index(drop=True)
)


# =========================================================
# 숫자형 데이터
# =========================================================

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


# =========================================================
# 날짜 숫자 변환
# =========================================================

BASE_DATE = pd.Timestamp("2000-01-01")

movies["openDt_date"] = pd.to_datetime(
    movies["openDt"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

movies["first_date_date"] = pd.to_datetime(
    movies["first_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

movies["openDt_num"] = (
    movies["openDt_date"] - BASE_DATE
).dt.days

movies["first_date_num"] = (
    movies["first_date_date"] - BASE_DATE
).dt.days


# =========================================================
# 제목
# =========================================================

st.title("🎬 영화 흥행 예측기")

st.write(
    "KOBIS 영화 데이터를 이용하여 "
    "**영화의 총 관객 수를 다중 회귀로 예측**합니다."
)


# =========================================================
# 데이터 정보
# =========================================================

st.subheader("📊 데이터 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "전체 영화",
    f"{len(movies):,}편"
)

col2.metric(
    "일별 데이터 기간",
    f"{period_start:%Y-%m-%d} ~ {period_end:%Y-%m-%d}"
)

col3.metric(
    "영화 정보 행",
    f"{len(movies):,}편"
)

col4.metric(
    "영화 정보 열",
    f"{len(movies.columns) - 3}개"
)


# =========================================================
# 영화별 표
# =========================================================

st.subheader("📋 영화별 데이터")

st.caption(
    "영화코드(movieCd) 순으로 정렬한 영화별 데이터입니다."
)

# 원본의 첫 번째 데이터 행을 그대로 표시
st.dataframe(
    original_top_row,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 변수 선택
# =========================================================

st.subheader("⚙️ 예측 변수 선택")

st.write(
    "체크한 변수만 다중 회귀 모델의 입력 변수로 사용합니다."
)


variable_names = {
    "openDt": "개봉일",
    "genre": "장르",
    "nation": "국가",
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "first_date": "10위권 첫 등장일",
    "peak": "성수기 개봉 여부",
    "first_week_audi": "첫 주 관객",
    "days_in_top10": "10위권 유지일"
}


variable_columns = list(variable_names.keys())

selected_variables = []

check_cols = st.columns(3)

for i, variable in enumerate(variable_columns):

    with check_cols[i % 3]:

        checked = st.checkbox(
            variable_names[variable],
            value=True,
            key=f"variable_{variable}"
        )

        if checked:
            selected_variables.append(variable)


if not selected_variables:

    st.warning(
        "예측 변수를 하나 이상 선택해주세요."
    )

    st.stop()


# =========================================================
# 실제 모델용 열 이름
# =========================================================

feature_map = {
    "openDt": "openDt_num",
    "genre": "genre",
    "nation": "nation",
    "first_scrn": "first_scrn",
    "first_show": "first_show",
    "first_date": "first_date_num",
    "peak": "peak",
    "first_week_audi": "first_week_audi",
    "days_in_top10": "days_in_top10"
}

model_features = [
    feature_map[x]
    for x in selected_variables
]


# =========================================================
# 테스트 영화 선정
#
# 영화코드 순으로 정렬된 상태에서
# 10편마다 앞의 3편을 테스트용으로 사용
# =========================================================

test_indices = []

for start in range(
    0,
    len(movies),
    10
):

    block = list(
        range(
            start,
            min(
                start + 10,
                len(movies)
            )
        )
    )

    # 각 10편 묶음의 앞 3편
    test_indices.extend(
        block[:3]
    )


test_indices = sorted(
    set(test_indices)
)


test_mask = movies.index.isin(
    test_indices
)


train_df = movies.loc[
    ~test_mask
].copy()

test_df = movies.loc[
    test_mask
].copy()


# =========================================================
# X / y
# =========================================================

X_train = train_df[
    model_features
].copy()

X_test = test_df[
    model_features
].copy()

y_train = pd.to_numeric(
    train_df["total_audi"],
    errors="coerce"
)

y_test = pd.to_numeric(
    test_df["total_audi"],
    errors="coerce"
)


# =========================================================
# 숫자형 / 범주형 변수 구분
# =========================================================

numeric_features = []
categorical_features = []

for col in model_features:

    if pd.api.types.is_numeric_dtype(
        X_train[col]
    ):

        numeric_features.append(col)

    else:

        categorical_features.append(col)


# =========================================================
# 숫자형 변수 안전 처리
# =========================================================

for col in numeric_features:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )

    # inf 제거
    X_train[col] = X_train[col].replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test[col] = X_test[col].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # 학습 데이터 중앙값
    median = X_train[col].median()

    if pd.isna(median):
        median = 0

    X_train[col] = X_train[col].fillna(
        median
    )

    X_test[col] = X_test[col].fillna(
        median
    )


# =========================================================
# 범주형 변수 안전 처리
# =========================================================

for col in categorical_features:

    X_train[col] = (
        X_train[col]
        .astype(str)
        .replace("nan", "알 수 없음")
    )

    X_test[col] = (
        X_test[col]
        .astype(str)
        .replace("nan", "알 수 없음")
    )


# =========================================================
# 원-핫 인코딩
# =========================================================

if categorical_features:

    combined = pd.concat(
        [
            X_train,
            X_test
        ],
        axis=0
    )

    combined = pd.get_dummies(
        combined,
        columns=categorical_features,
        dtype=float
    )

    X_train_final = combined.iloc[
        :len(X_train)
    ].copy()

    X_test_final = combined.iloc[
        len(X_train):
    ].copy()

else:

    X_train_final = X_train.copy()
    X_test_final = X_test.copy()


# =========================================================
# 최종 숫자 안전 처리
# =========================================================

X_train_final = (
    X_train_final
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
    .astype(float)
)

X_test_final = (
    X_test_final
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
    .astype(float)
)


# =========================================================
# 목표값이 없는 학습 영화 제거
# =========================================================

valid_train = y_train.notna()

X_train_final = X_train_final.loc[
    valid_train
]

y_train = y_train.loc[
    valid_train
]


# =========================================================
# 다중 회귀 모델
# =========================================================

model = LinearRegression()

model.fit(
    X_train_final,
    y_train
)


# =========================================================
# 테스트 영화 예측
# =========================================================

predictions = model.predict(
    X_test_final
)

# 관객 수는 음수가 될 수 없으므로 0으로 제한
predictions = np.maximum(
    predictions,
    0
)


# =========================================================
# 평가
# =========================================================

valid_test = y_test.notna()

actual = y_test.loc[
    valid_test
]

predicted = predictions[
    valid_test.to_numpy()
]


if len(actual) >= 2:

    r2 = r2_score(
        actual,
        predicted
    )

else:

    r2 = np.nan


mae = mean_absolute_error(
    actual,
    predicted
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)


# =========================================================
# 결과 요약
# =========================================================

st.subheader("📈 학습 및 평가 결과")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "전체 영화",
    f"{len(movies):,}편"
)

c2.metric(
    "학습에 사용한 영화",
    f"{len(train_df):,}편"
)

c3.metric(
    "평가한 영화",
    f"{len(test_df):,}편"
)

c4.metric(
    "R² 점수",
    f"{r2:.4f}" if not np.isnan(r2) else "계산 불가"
)


st.info(
    f"**기준 기간:** "
    f"{period_start:%Y-%m-%d} ~ "
    f"{period_end:%Y-%m-%d}"
)


m1, m2 = st.columns(2)

m1.metric(
    "MAE",
    f"{mae:,.0f}명"
)

m2.metric(
    "RMSE",
    f"{rmse:,.0f}명"
)


# =========================================================
# 테스트 결과 표
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

result["오차율 (%)"] = np.where(
    result["total_audi"] != 0,
    (
        result["오차"]
        / result["total_audi"]
        * 100
    ),
    np.nan
)

result = (
    result
    .sort_values("movieCd")
    .reset_index(drop=True)
)


st.subheader("🎞️ 테스트 영화 예측 결과")

display_result = result.rename(
    columns={
        "movieCd": "영화코드",
        "movieNm": "영화명",
        "total_audi": "실제 총 관객 수"
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

low_mask = (
    result["예측 총 관객 수"] < 1000
)

low_count = int(
    low_mask.sum()
)


# =========================================================
# 산점도
# =========================================================

st.subheader(
    "📊 실제 총 관객 수 vs 예측 총 관객 수"
)

st.caption(
    "가로축: 실제 총 관객 수 / "
    "세로축: 예측 총 관객 수 / "
    "두 축 모두 로그 스케일"
)


fig = go.Figure()


# ---------------------------------------------------------
# 일반 테스트 영화
# ---------------------------------------------------------

normal = result.loc[
    (~low_mask)
    & (result["total_audi"] > 0)
    & (result["예측 총 관객 수"] > 0)
]


if len(normal) > 0:

    fig.add_trace(
        go.Scatter(
            x=normal["total_audi"],
            y=normal["예측 총 관객 수"],
            mode="markers",
            name="테스트 영화",
            text=normal["movieNm"],
            customdata=np.column_stack(
                [
                    normal["movieCd"],
                    normal["total_audi"],
                    normal["예측 총 관객 수"],
                    normal["오차"]
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제: %{customdata[1]:,.0f}명<br>"
                "예측: %{customdata[2]:,.0f}명<br>"
                "오차: %{customdata[3]:,.0f}명"
                "<extra></extra>"
            ),
            marker={
                "size": 9
            }
        )
    )


# ---------------------------------------------------------
# 1,000명 미만 영화
# ---------------------------------------------------------

low = result.loc[
    low_mask
    & (result["total_audi"] > 0)
]


if len(low) > 0:

    # 로그 그래프의 바닥 부분에 표시
    floor_value = 900

    fig.add_trace(
        go.Scatter(
            x=low["total_audi"],
            y=[
                floor_value
            ] * len(low),
            mode="markers",
            name="예측 1,000명 미만",
            text=low["movieNm"],
            customdata=np.column_stack(
                [
                    low["movieCd"],
                    low["total_audi"],
                    low["예측 총 관객 수"],
                    low["오차"]
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제: %{customdata[1]:,.0f}명<br>"
                "예측: %{customdata[2]:,.0f}명<br>"
                "오차: %{customdata[3]:,.0f}명"
                "<extra></extra>"
            ),
            marker={
                "symbol": "triangle-down",
                "size": 11
            }
        )
    )


# ---------------------------------------------------------
# 실제값 = 예측값 대각선
# ---------------------------------------------------------

positive = result.loc[
    (result["total_audi"] > 0)
    & (result["예측 총 관객 수"] > 0)
]


if len(positive) > 0:

    line_min = min(
        positive["total_audi"].min(),
        positive["예측 총 관객 수"].min()
    )

    line_max = max(
        positive["total_audi"].max(),
        positive["예측 총 관객 수"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[
                line_min,
                line_max
            ],
            y=[
                line_min,
                line_max
            ],
            mode="lines",
            name="실제값 = 예측값",
            line={
                "dash": "dash"
            }
        )
    )


# =========================================================
# 그래프 설정
# =========================================================

fig.update_layout(
    height=650,

    xaxis={
        "title": "실제 총 관객 수",
        "type": "log"
    },

    yaxis={
        "title": "예측 총 관객 수",
        "type": "log"
    },

    hovermode="closest"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 1,000명 미만 개수
# =========================================================

st.info(
    f"예측 총 관객 수가 1,000명보다 작게 나온 영화는 "
    f"**{low_count}편**입니다."
)


# =========================================================
# 선택된 변수
# =========================================================

with st.expander("🔎 현재 모델에 사용한 변수"):

    for variable in selected_variables:

        st.write(
            f"• {variable_names[variable]}"
        )


# =========================================================
# 전체 영화 데이터
# =========================================================

with st.expander("📚 전체 영화별 데이터 보기"):

    st.dataframe(
        movies_original,
        use_container_width=True,
        hide_index=True
    )
