# app.py
import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import rc
import matplotlib.font_manager as fm


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

## 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# 깃허브 리눅스 기준
if platform.system() == 'Linux':
    fontname = './NanumGothic.ttf'
    font_files = fm.findSystemFonts(fontpaths=fontname)
    fm.fontManager.addfont(fontname)
    fm._load_fontmanager(try_read_cache=False)
    rc('font', family='NanumGothic')

# ------------------------------------------------
# 페이지 기본 설정
# ------------------------------------------------
st.set_page_config(
    page_title="금 · 환율 데이터 분석 & 회귀 모델 성능 비교",
    page_icon="💹",
    layout="wide"
)

st.title("💹 금 · 환율 변동 추이에 따른 화폐가치")

# ------------------------------------------------
# 데이터 로딩 함수
# ------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('금환율_병합데이터_FX기간만 (1).csv')  # CSV 파일 로드 시도
        return df                     # 성공 시 DataFrame 반환
    except FileNotFoundError:
        st.error("🚨 파일이 존재하지 않습니다. ")
        return pd.DataFrame()         # 실패 시 빈 DataFrame 반환


def get_basic_info(df: pd.DataFrame):
    return {
        "행 개수": df.shape[0],
        "열 개수": df.shape[1],
        "결측치 총합": int(df.isna().sum().sum()),
        "중복 행 개수": int(df.duplicated().sum())
    }

# 단일 변수 시각화
def plot_univariate(df, col, chart_type):
    series = df[col].dropna()
    fig, ax = plt.subplots()

    if chart_type == "히스토그램":
        ax.hist(series, bins=20)
        ax.set_title(f"{col} 분포 (히스토그램)")
        ax.set_xlabel(col)
        ax.set_ylabel("빈도")

    elif chart_type == "박스플롯":
        ax.boxplot(series, vert=True)
        ax.set_title(f"{col} 분포 (박스플롯)")
        ax.set_ylabel(col)

    elif chart_type == "선그래프":
        ax.plot(series.values)
        ax.set_title(f"{col} 추이 (선그래프)")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)

    st.pyplot(fig)

# 상관관계 히트맵
def plot_corr_heatmap(df):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        st.warning("상관관계 히트맵을 그리려면 숫자형 컬럼이 2개 이상 필요합니다.")
        return

    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("숫자형 변수 상관관계 히트맵")
    st.pyplot(fig)

# 회귀 모델 성능 지표 계산
def get_regression_metrics(y_true, y_pred, model_name="model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "모델": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

# 막대 그래프 그리기
def plot_metric_bar(df_metrics, metric_name):
    fig, ax = plt.subplots()
    ax.bar(df_metrics["모델"], df_metrics[metric_name])
    ax.set_title(f"모델별 {metric_name} 비교")
    ax.set_ylabel(metric_name)
    ax.set_xticklabels(df_metrics["모델"], rotation=20)
    st.pyplot(fig)

# ------------------------------------------------
# 1. 파일 업로드
# ------------------------------------------------

df = load_data()

if df.empty:
    st.stop()

# ------------------------------------------------
# 2. 데이터 기본 정보
# ------------------------------------------------
st.header("📖 데이터 기본 정보")

info = get_basic_info(df)
col_info1, col_info2 = st.columns([1, 2])

with col_info1:
    st.subheader("데이터 요약")
    st.write(info)

    st.subheader("컬럼 · 데이터 타입")
    type_df = pd.DataFrame({
        "컬럼명": df.columns,
        "dtype": df.dtypes.astype(str)
    })
    st.dataframe(type_df)

with col_info2:
    st.subheader("숫자형 컬럼 기술통계")
    num_desc = df.select_dtypes(include=np.number).describe()
    st.dataframe(num_desc)

    st.subheader("컬럼별 결측치 개수")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.write("결측치가 없습니다.")
    else:
        st.dataframe(missing.to_frame("결측치 개수"))

st.subheader("데이터 미리보기 (상위 20행)")
st.dataframe(df.head(20))

# ------------------------------------------------
# 3. 주요 지표 시각화 (차트 선택)
# ------------------------------------------------
st.header("📶 주요 지표 시각화")

num_cols = df.select_dtypes(include=np.number).columns.tolist()

vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    st.subheader("단일 변수 분포 시각화")
    if len(num_cols) == 0:
        st.warning("숫자형 컬럼이 없습니다.")
    else:
        selected_col = st.selectbox("시각화할 숫자형 컬럼 선택", num_cols)
        chart_type = st.radio(
            "차트 유형 선택",
            ["히스토그램", "박스플롯", "선그래프"],
            horizontal=True
        )
        plot_univariate(df, selected_col, chart_type)

with vis_col2:
    st.subheader("상관관계 히트맵")
    if st.button("상관관계 히트맵 그리기"):
        plot_corr_heatmap(df)

st.subheader("산점도(Scatter) 시각화")
if len(num_cols) >= 2:
    scatter_col1, scatter_col2 = st.columns(2)
    with scatter_col1:
        x_col = st.selectbox("X축 컬럼", num_cols, key="scatter_x")
    with scatter_col2:
        y_col = st.selectbox("Y축 컬럼", num_cols, key="scatter_y")

    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col], alpha=0.7)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col} 산점도")
    st.pyplot(fig)
else:
    st.info("산점도를 위해서는 숫자형 컬럼이 2개 이상 필요합니다.")

# ------------------------------------------------
# 4. 회귀 모델 성능 비교
# ------------------------------------------------
st.header("🔎 회귀 모델 성능 비교")

st.markdown("업로드한 데이터에서 **목표 변수(타깃)** 를 선택해 회귀 모델을 비교합니다.")

if len(num_cols) < 2:
    st.warning("회귀 모델 비교를 위해서는 숫자형 컬럼이 최소 2개 이상 필요합니다.")
    st.stop()

# 타깃 컬럼 기본값: '달러(원)' 이 있으면 우선 사용, 없으면 마지막 숫자형 컬럼
if "달러(원)" in num_cols:
    default_target_idx = num_cols.index("달러(원)")
else:
    default_target_idx = len(num_cols) - 1

target_col = st.selectbox(
    "목표 변수(타깃) 컬럼 선택",
    num_cols,
    index=default_target_idx
)

# 피처 컬럼 선택 (타깃 제외 숫자형)
feature_candidates = [c for c in num_cols if c != target_col]

feature_cols = st.multiselect(
    "설명 변수(피처)로 사용할 컬럼 선택",
    feature_candidates,
    default=feature_candidates
)

if len(feature_cols) == 0:
    st.warning("최소 1개 이상의 피처 컬럼을 선택해야 합니다.")
    st.stop()

X = df[feature_cols]
y = df[target_col]

# 결측치 제거
data_all = pd.concat([X, y], axis=1).dropna()
X_clean = data_all[feature_cols]
y_clean = data_all[target_col]

if X_clean.empty:
    st.error("결측치 제거 후 남은 데이터가 없습니다. 데이터를 확인해 주세요.")
    st.stop()

st.subheader("학습 / 평가 설정")

opt1, opt2, opt3 = st.columns(3)
with opt1:
    test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, step=0.05)
with opt2:
    random_state = st.number_input("random_state", 0, 9999, 42)
with opt3:
    use_scaler = st.checkbox("표준화(StandardScaler) 사용", value=True)

st.subheader("비교할 회귀 모델 선택")
model_names = st.multiselect(
    "모델 선택",
    ["Linear Regression", "Random Forest", "KNN Regressor", "SVR"],
    default=["Linear Regression", "Random Forest"]
)

if st.button("회귀 모델 학습 및 성능 평가 실행"):
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )

    model_dict = {}
    if "Linear Regression" in model_names:
        model_dict["Linear Regression"] = LinearRegression()
    if "Random Forest" in model_names:
        model_dict["Random Forest"] = RandomForestRegressor(
            n_estimators=300, random_state=random_state
        )
    if "KNN Regressor" in model_names:
        model_dict["KNN Regressor"] = KNeighborsRegressor(n_neighbors=5)
    if "SVR" in model_names:
        model_dict["SVR"] = SVR(kernel="rbf")

    if not model_dict:
        st.warning("최소 1개 이상의 모델을 선택해 주세요.")
        st.stop()

    results = []
    for name, model in model_dict.items():
        # 스케일링이 필요한 모델들에 대해 파이프라인 구성
        if use_scaler:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
        else:
            pipe = Pipeline([
                ("model", model)
            ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = get_regression_metrics(y_test, y_pred, model_name=name)
        results.append(metrics)

    metrics_df = pd.DataFrame(results)

    st.subheader("모델 성능 비교 표")
    st.dataframe(
        metrics_df.set_index("모델").style.format({
            "MSE": "{:.3f}",
            "RMSE": "{:.3f}",
            "MAE": "{:.3f}",
            "R2": "{:.3f}"
        })
    )

    st.subheader("모델 성능 비교 그래프")
    metric_choice = st.selectbox(
        "그래프로 확인할 지표 선택",
        ["RMSE", "MAE", "R2"]
    )
    plot_metric_bar(metrics_df, metric_choice)

else:
    st.info("아래 버튼을 눌러 회귀 모델을 학습하고 성능을 비교해 보세요.")
