import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.font_manager as fm
import platform
from matplotlib import rc



# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 깃허브 리눅스 기준
if platform.system() == 'Linux':
    fontname = './NanumGothic.ttf'
    font_files = fm.findSystemFonts(fontpaths=fontname)
    fm.fontManager.addfont(fontname)
    fm._load_fontmanager(try_read_cache=False)
    rc('font', family='NanumGothic')
    
# ---------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------
st.set_page_config(page_title="데이터 분석 웹앱", layout="wide")
st.title("📊 데이터 분석 웹앱 (Streamlit)")

# ---------------------------------------------
# 파일 업로드
# ---------------------------------------------
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv", "xlsx"])

# ---------------------------------------------
# 파일 읽기
# ---------------------------------------------
if uploaded_file is not None:
    # 확장자에 따라 읽기
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("파일 업로드 성공!")
    
    # ---------------------------------------------
    # 데이터프레임 출력
    # ---------------------------------------------
    st.subheader("📁 데이터 미리보기")
    st.dataframe(df)

    # ---------------------------------------------
    # 기본 정보
    # ---------------------------------------------
    st.subheader("📌 데이터 기본 정보")
    st.write("행(Row) 수:", df.shape[0])
    st.write("열(Column) 수:", df.shape[1])

    # 통계 요약
    st.subheader("📈 기술통계 요약")
    st.dataframe(df.describe())

    # ---------------------------------------------
    # 시각화 설정
    # ---------------------------------------------
    st.subheader("📊 시각화 차트 만들기")

    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols) >= 1:
        chart_type = st.selectbox("차트 종류를 선택하세요:", ["히스토그램", "라인차트", "박스플롯", "바차트(scatter 포함)"])
        selected_col = st.selectbox("수치형 컬럼 선택:", numeric_cols)

        # ---------------------------------------------
        # 시각화 생성
        # ---------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4))

        if chart_type == "히스토그램":
            sns.histplot(df[selected_col], kde=True, ax=ax)
            ax.set_title(f"{selected_col} 히스토그램")

        elif chart_type == "라인차트":
            ax.plot(df[selected_col])
            ax.set_title(f"{selected_col} 라인차트")

        elif chart_type == "박스플롯":
            sns.boxplot(x=df[selected_col], ax=ax)
            ax.set_title(f"{selected_col} 박스플롯")

        elif chart_type == "바차트(scatter 포함)":
            x_col = st.selectbox("X축 컬럼 선택:", df.columns)
            y_col = st.selectbox("Y축 컬럼 선택:", numeric_cols)
            ax.scatter(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col} 산점도")

        st.pyplot(fig)

    else:
        st.warning("수치형 컬럼이 없어서 차트를 생성할 수 없습니다.")

else:
    st.info("데이터를 업로드하면 분석이 시작됩니다.")
