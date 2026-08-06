"""
Golden Batch 기반 페니실린 공정 모니터링 대시보드 - 1단계 스켈레톤

구성 (체크리스트 4, 5번 반영):
    1. 제목
    2. 배치 선택창
    3. 선택 배치 KPI 카드 4개
    4. 전략별(RC/OC/APC/Fault) 성능 비교 박스플롯
    5. 골든밴드 / 농도 예측 결과 - 자리만 표시 (추후 로직 연결)

실행 방법:
    streamlit run app.py

필요 파일 (같은 폴더에 위치):
    - batch_summary.csv   (Batch_ID, Strategy, total_yield, duration_h, final_conc, productivity)
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Penicillin Golden Batch Dashboard", layout="wide")

STRATEGY_ORDER = ["RC", "OC", "APC", "Fault"]
STRATEGY_COLORS = {
    "RC": "#5DCAA5",
    "OC": "#7F77DD",
    "APC": "#378ADD",
    "Fault": "#E24B4A",
}


@st.cache_data
def load_batch_summary(path: str = "batch_summary.csv") -> pd.DataFrame:
    """배치당 한 행으로 요약된 KPI 데이터를 불러온다."""
    base_path = Path(__file__).resolve().parent
    csv_path = base_path / path
    if not csv_path.exists():
        raise FileNotFoundError(f"batch_summary.csv를 찾을 수 없습니다: {csv_path}")
    return pd.read_csv(csv_path)


def build_strategy_boxplots(agg: pd.DataFrame) -> go.Figure:
    """RC/OC/APC/Fault 전략별 총회수량·최종농도·생산성 박스플롯 3개를 만든다."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("총회수량 (kg)", "최종 페니실린 농도 (g/L)", "시간당 생산성 (kg/h)"),
    )

    metrics = [("total_yield", 1), ("final_conc", 2), ("productivity", 3)]

    for metric, col in metrics:
        for strategy in STRATEGY_ORDER:
            sub = agg[agg["Strategy"] == strategy]
            fig.add_trace(
                go.Box(
                    y=sub[metric],
                    name=strategy,
                    marker_color=STRATEGY_COLORS[strategy],
                    showlegend=(col == 1),
                    legendgroup=strategy,
                ),
                row=1,
                col=col,
            )

    fig.update_layout(height=420, boxmode="group", margin=dict(t=60, b=20))
    return fig


def main() -> None:
    st.title("Golden Batch 기반 페니실린 공정 모니터링")
    st.caption("고수율 배치(Golden Batch) 기준 이탈 진단 + 미래 페니실린 농도 예측")

    agg = load_batch_summary()

    # 2. 전략 선택 + 배치 슬라이더
    strategy = st.segmented_control("제어전략", STRATEGY_ORDER, default=STRATEGY_ORDER[0])
    filtered = agg[agg["Strategy"] == strategy]
    batch_list = sorted(filtered["Batch_ID"].unique())

    if not batch_list:
        st.error("선택한 전략에 해당하는 배치가 없습니다.")
        return

    selected_batch = st.select_slider("배치 선택", options=batch_list, value=batch_list[0])
    row = filtered[filtered["Batch_ID"] == selected_batch].iloc[0]

    st.caption(f"선택한 배치의 제어전략: **{row['Strategy']}**")

    # 3. 선택 배치 KPI 카드
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총회수량", f"{row['total_yield']:,.0f} kg")
    col2.metric("운전시간", f"{row['duration_h']:.1f} h")
    col3.metric("최종 농도", f"{row['final_conc']:.2f} g/L")
    col4.metric("시간당 생산성", f"{row['productivity']:,.0f} kg/h")

    st.divider()

    # 4. 전략별 성능 비교 박스플롯
    st.subheader("전략별 성능 비교 (RC / OC / APC / Fault)")
    st.plotly_chart(build_strategy_boxplots(agg), use_container_width=True)

    st.divider()

    # 5. 골든밴드 / 예측 결과 - 자리만 표시
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("골든배치 기준선 비교")
        st.info("추후 골든배치 로직 완성 후 이 자리에 궤적 비교 차트를 연결할 예정입니다.")

    with col_right:
        st.subheader("농도 예측 결과")
        st.info("추후 예측 모델 완성 후 이 자리에 실제 vs 예측 농도 차트를 연결할 예정입니다.")


if __name__ == "__main__":
    main()