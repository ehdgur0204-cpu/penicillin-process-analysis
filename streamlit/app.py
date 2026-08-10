"""Streamlit dashboard entry point."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# 프로젝트의 src 폴더 연결
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from predict_12h import predict_12h


# Streamlit 설정
st.set_page_config(
    page_title="Penicillin Process Analysis",
    page_icon="🧪",
    layout="wide",
)


# 데이터 불러오기
DATA_PATH = ROOT_DIR / "data" / "raw" / "merged_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# 화면
st.title("🧪 Penicillin Process Analysis")
st.write("페니실린 생산 공정 분석 대시보드입니다.")

st.divider()

st.subheader("📈 12시간 후 페니실린 농도 예측")


# Batch 선택
batch_ids = sorted(
    df["Batch_ID"].dropna().unique()
)

selected_batch = st.selectbox(
    "Batch 선택",
    batch_ids,
)


# 선택한 Batch 데이터
batch_df = df[
    df["Batch_ID"] == selected_batch
].sort_values("Time (h)")


min_time = float(
    batch_df["Time (h)"].min()
)

max_time = float(
    batch_df["Time (h)"].max()
)


# 현재 시간 선택
current_time = st.number_input(
    "현재 공정 시간 (h)",
    min_value=min_time,
    max_value=max_time,
    value=min_time,
    step=0.2,
)


# 예측 버튼
if st.button(
    "🔮 12시간 후 농도 예측",
    type="primary",
):

    result = predict_12h(
        df=df,
        batch_id=int(selected_batch),
        current_time=float(current_time),
    )

    st.success("12시간 후 농도 예측 완료")


    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric(
            "현재 농도",
            f"{result['current_p']:.2f} g/L",
        )


    with col2:
        st.metric(
            "12시간 후 예측 농도",
            f"{result['predicted_p_12h']:.2f} g/L",
        )


    with col3:
        st.metric(
            "예상 변화량",
            f"{result['change']:+.2f} g/L",
        )


    st.write(
        f"현재 시점: **{result['current_time']:.1f} h**"
    )

    st.write(
        f"예측 시점: **{result['target_time']:.1f} h**"
    )


