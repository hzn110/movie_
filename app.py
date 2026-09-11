import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================================================
# 설정
# =========================================================

st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)

MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8-sig")
    daily = pd.read_csv(DAILY_URL, encoding="utf-8-sig")
    return movies, daily


try:
    movies, daily = load_data()
except Exception as e:
    st.error("CSV 데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 기본 전처리
# =========================================================

movies.columns = movies.columns.str.strip()
daily.columns = daily.columns.str.strip()

movies["movieCd"] = movies["movieCd"].astype(str).str.strip()
daily["영화코드"] = daily["영화코드"].astype(str).str.strip()

# 숫자형
for col in [
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]:
    movies[col] = pd.to_numeric(movies[col], errors="coerce")

# 날짜
movies["openDt"] = pd.to_datetime(
    movies["openDt"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

daily["날짜"] = pd.to_datetime(
    daily["날짜"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

# 기간
period_start = daily["날짜"].min()
period_end = daily["날짜"].max()

# 영화코드 순 정렬
movies = movies.sort_values("movieCd").reset_index(drop=True)


# =========================================================
# 화면
# =========================================================

st.title("🎬 영화 흥행 예측기")

st.write(
    "영화별 KOBIS 데이터를 이용하여 영화의 총 관객 수를 "
    "다중 회귀 모델로 예측합니다."
)


# =========================================================
# 데이터 정보
# =========================================================

st.subheader("📊 데이터 정보")

c1, c2, c3 = st.columns(3)

c1.metric(
    "전체 영화",
    f"{len(movies):,}편"
)

c2.metric(
    "일별 데이터 기간",
    f"{period_start:%Y-%m-%d} ~ {period_end:%Y-%m-%d}"
)

c3.metric(
    "영화 정보 열",
    f"{len(movies.columns)}개"
)


# =========================================================
# 영화별 표 맨 위
# =========================================================

st.subheader("📋 영화별 데이터")

st.caption(
    "영화코드(movieCd) 순으로 정렬한 영화별 데이터입니다."
)

st.dataframe(
    movies.head(1),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 변수 선택
# =========================================================

st.subheader("⚙️ 예측 변수 선택")

labels = {
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

features = list(labels.keys())

selected = []

cols = st.columns(3)

for i, feature in enumerate(features):
    with cols[i % 3]:
        if st.checkbox(
            labels[feature],
            value=True,
            key=f"check_{feature}"
        ):
            selected.append(feature)

if not selected:
    st.warning("예측 변수를 하나 이상 선택해주세요.")
    st.stop()


# =========================================================
# 모델용 데이터 생성
# =========================================================

df = movies.copy()

# 날짜 → 숫자
base_date = pd.Timestamp("2000-01-01")

df["openDt_num"] = (
    df["openDt"] - base_date
).dt.days

df["first_date_num"] = pd.to_datetime(
    df["first_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

df["first_date_num"] = (
    df["first_date_num"] - base_date
).dt.days


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
    for x in selected
]


# =========================================================
# 10편마다 앞 3편 테스트
# =========================================================

test_indices = []

for start in range(0, len(df), 10):
    block = list(
        range(
            start,
            min(start + 10, len(df))
        )
    )

    # 앞의 3편
    test_indices.extend(block[:3])

test_indices = sorted(set(test_indices))

test_mask = df.index.isin(test_indices)

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()


# =========================================================
# X / y
# =========================================================

X_train = train_df[model_features].copy()
X_test = test_df[model_features].copy()

y_train = train_df["total_audi"].copy()
y_test = test_df["total_audi"].copy()


# =========================================================
# 자료형 구분
# =========================================================

categorical = []
numeric = []

for col in model_features:
    if X_train[col].dtype == "object":
        categorical.append(col)
    else:
        numeric.append(col)


# =========================================================
# 결측치 + 원핫 인코딩
# =========================================================

transformers = []

if numeric:

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

    X_train_numeric = numeric_imputer.fit_transform(
        X_train[numeric]
    )

    X_test_numeric = numeric_imputer.transform(
        X_test[numeric]
    )

    X_train_numeric = pd.DataFrame(
        X_train_numeric,
        columns=numeric,
        index=X_train.index
    )

    X_test_numeric = pd.DataFrame(
        X_test_numeric,
        columns=numeric,
        index=X_test.index
    )

else:

    X_train_numeric = pd.DataFrame(
        index=X_train.index
    )

    X_test_numeric = pd.DataFrame(
        index=X_test.index
    )


if categorical:

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    train_cat = categorical_imputer.fit_transform(
        X_train[categorical]
    )

    test_cat = categorical_imputer.transform(
        X_test[categorical]
    )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    train_cat = encoder.fit_transform(train_cat)
    test_cat = encoder.transform(test_cat)

    cat_columns = encoder.get_feature_names_out(
        categorical
    )

    X_train_cat = pd.DataFrame(
        train_cat,
        columns=cat_columns,
        index=X_train.index
    )

    X_test_cat = pd.DataFrame(
        test_cat,
        columns=cat_columns,
        index=X_test.index
    )

else:

    X_train_cat = pd.DataFrame(
        index=X_train.index
    )

    X_test_cat = pd.DataFrame(
        index=X_test.index
    )


# =========================================================
# 최종 학습 데이터
# =========================================================

X_train_final = pd.concat(
    [
        X_train_numeric,
        X_train_cat
    ],
    axis=1
)

X_test_final = pd.concat(
    [
        X_test_numeric,
        X_test_cat
    ],
    axis=1
)

# 혹시 남은 무한대 제거
X_train_final = X_train_final.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test_final = X_test_final.replace(
    [np.inf, -np.inf],
    np.nan
)

# 최종 결측치
X_train_final = X_train_final.fillna(0)
X_test_final = X_test_final.fillna(0)


# =========================================================
# 회귀 모델
# =========================================================

model = LinearRegression()

model.fit(
    X_train_final,
    y_train
)


# =========================================================
# 예측
# =========================================================

pred = model.predict(X_test_final)

# 음수 관객 수 방지
pred = np.maximum(pred, 0)


# =========================================================
# 평가
# =========================================================

score_r2 = r2_score(
    y_test,
    pred
)

score_mae = mean_absolute_error(
    y_test,
    pred
)

score_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)


# =========================================================
# 학습 / 평가 정보
# =========================================================

st.subheader("📈 모델 결과")

a, b, c, d = st.columns(4)

a.metric(
    "전체 영화",
    f"{len(df):,}편"
)

b.metric(
    "학습 영화",
    f"{len(train_df):,}편"
)

c.metric(
    "평가 영화",
    f"{len(test_df):,}편"
)

d.metric(
    "R² 점수",
    f"{score_r2:.4f}"
)

st.write(
    f"**기준 기간:** "
    f"{period_start:%Y-%m-%d} ~ {period_end:%Y-%m-%d}"
)

e, f = st.columns(2)

e.metric(
    "MAE",
    f"{score_mae:,.0f}명"
)

f.metric(
    "RMSE",
    f"{score_rmse:,.0f}명"
)


# =========================================================
# 테스트 결과
# =========================================================

result = test_df[
    [
        "movieCd",
        "movieNm",
        "total_audi"
    ]
].copy()

result["예측 총 관객 수"] = pred

result["오차"] = (
    result["예측 총 관객 수"]
    - result["total_audi"]
)

result["절대 오차"] = (
    result["오차"].abs()
)

result["오차율"] = np.where(
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

show_result = result.rename(
    columns={
        "movieCd": "영화코드",
        "movieNm": "영화명",
        "total_audi": "실제 총 관객 수",
        "오차율": "오차율 (%)"
    }
)

st.dataframe(
    show_result.style.format({
        "실제 총 관객 수": "{:,.0f}",
        "예측 총 관객 수": "{:,.0f}",
        "오차": "{:,.0f}",
        "절대 오차": "{:,.0f}",
        "오차율 (%)": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 1,000명 미만
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
    "📊 실제 관객 수와 예측 관객 수"
)

st.caption(
    "가로축 = 실제 총 관객 수 / "
    "세로축 = 예측 총 관객 수 / "
    "두 축 모두 로그 스케일"
)

fig = go.Figure()


# 일반 영화
normal = result.loc[
    ~low_mask &
    (result["total_audi"] > 0) &
    (result["예측 총 관객 수"] > 0)
]

if len(normal) > 0:

    fig.add_trace(
        go.Scatter(
            x=normal["total_audi"],
            y=normal["예측 총 관객 수"],
            mode="markers",
            name="테스트 영화",
            text=normal["movieNm"],
            customdata=np.column_stack([
                normal["movieCd"],
                normal["total_audi"],
                normal["예측 총 관객 수"],
                normal["오차"]
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제: %{customdata[1]:,.0f}명<br>"
                "예측: %{customdata[2]:,.0f}명<br>"
                "오차: %{customdata[3]:,.0f}명"
                "<extra></extra>"
            ),
            marker=dict(size=9)
        )
    )


# 1,000명 미만
low = result.loc[
    low_mask &
    (result["total_audi"] > 0)
]

if len(low) > 0:

    # 그래프 바닥에 붙여 표시
    floor = 900

    fig.add_trace(
        go.Scatter(
            x=low["total_audi"],
            y=[floor] * len(low),
            mode="markers",
            name="예측 < 1,000명",
            text=low["movieNm"],
            customdata=np.column_stack([
                low["movieCd"],
                low["total_audi"],
                low["예측 총 관객 수"],
                low["오차"]
            ]),
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


# =========================================================
# y=x 대각선
# =========================================================

positive = result[
    (result["total_audi"] > 0) &
    (result["예측 총 관객 수"] > 0)
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
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            name="실제값 = 예측값",
            line=dict(dash="dash")
        )
    )


fig.update_layout(
    height=650,
    xaxis=dict(
        title="실제 총 관객 수",
        type="log"
    ),
    yaxis=dict(
        title="예측 총 관객 수",
        type="log"
    ),
    hovermode="closest"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.info(
    f"예측 총 관객 수가 1,000명보다 작게 나온 영화는 "
    f"**{low_count}편**입니다."
)


# =========================================================
# 선택 변수
# =========================================================

with st.expander("🔎 선택한 예측 변수"):
    for x in selected:
        st.write(f"- {labels[x]}")


# =========================================================
# 전체 데이터
# =========================================================

with st.expander("📚 전체 영화 데이터 보기"):
    st.dataframe(
        movies,
        use_container_width=True,
        hide_index=True
    )
