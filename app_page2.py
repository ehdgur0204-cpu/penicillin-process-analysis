import streamlit as st
import time

from data_loader import load_data
from replay_utils import get_replay_state


# =================================================
# 기본 설정
# =================================================
st.set_page_config(
    page_title="Penicillin Process Dashboard",
    page_icon="🧪",
    layout="wide",
)


# =================================================
# 공통 압축 CSS
# - 글씨는 너무 작게 줄이지 않고
# - 페이지/섹션/카드의 여백을 줄여 한 화면 밀도를 높임
# =================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 0.8rem;
        padding-left: 2.0rem;
        padding-right: 2.0rem;
    }

    h1 {
        font-size: 1.70rem !important;
        margin-top: 0 !important;
        margin-bottom: 0.10rem !important;
    }

    h2 {
        font-size: 1.30rem !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.20rem !important;
    }

    h3 {
        font-size: 1.05rem !important;
        margin-top: 0.20rem !important;
        margin-bottom: 0.15rem !important;
    }

    p {
        margin-bottom: 0.20rem;
    }

    hr {
        margin: 0.45rem 0 !important;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 0.65rem;
    }

    div[data-testid="stMetric"] {
        padding: 0.10rem 0;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
    }

    div[data-testid="stCaptionContainer"] p {
        font-size: 0.75rem !important;
        line-height: 1.25 !important;
        margin-bottom: 0 !important;
    }

    div[data-testid="stAlert"] {
        padding: 0.45rem 0.65rem;
    }

    .stButton > button {
        min-height: 2.0rem;
        padding: 0.25rem 0.65rem;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.0rem;
        padding-left: 1.0rem;
        padding-right: 1.0rem;
    }

    .app-title {
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.12rem;
    }

    .app-subtitle {
        font-size: 0.75rem;
        color: #6B7280;
        margin-bottom: 0.55rem;
    }

    .chip-wrap {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 0.42rem;
        flex-wrap: wrap;
        margin: 0.05rem 0 0.20rem 0;
    }

    .chip {
        border: 1px solid #D9DEE8;
        border-radius: 7px;
        padding: 0.34rem 0.58rem;
        font-size: 0.78rem;
        line-height: 1.15;
        white-space: nowrap;
        background: white;
    }

    .chip-label {
        color: #6B7280;
        margin-right: 0.32rem;
    }

    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.20rem 0.58rem;
        font-size: 0.78rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .status-normal {
        color: #238636;
        background: #E9F8EE;
        border: 1px solid #B7E4C7;
    }

    .status-caution {
        color: #9A6700;
        background: #FFF9E6;
        border: 1px solid #F5D77A;
    }

    .status-warning {
        color: #D1242F;
        background: #FFF0F0;
        border: 1px solid #FFB3B3;
    }

    .mini-meta {
        color: #6B7280;
        font-size: 0.76rem;
        line-height: 1.35;
    }

    .priority-title {
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 0.12rem;
    }

    .priority-body {
        font-size: 0.78rem;
        line-height: 1.35;
        color: #374151;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =================================================
# 데이터 로드
# =================================================
@st.cache_data
def get_dashboard_data():
    return load_data()


df = get_dashboard_data()


# =================================================
# 공통 타이틀
# =================================================
st.markdown(
    """
    <div class="app-title">페니실린 공정 모니터링 대시보드</div>
    <div class="app-subtitle">
        Test Batch Replay 기반 Golden Batch 이탈진단 및 12시간 후 페니실린 농도 예측
    </div>
    """,
    unsafe_allow_html=True,
)


# =================================================
# Session State / 페이지 이동
# =================================================
if "page" not in st.session_state:
    st.session_state["page"] = "공정 모니터링"

if "selected_batch" not in st.session_state:
    st.session_state["selected_batch"] = 91

if "replay_status" not in st.session_state:
    st.session_state["replay_status"] = "대기 중"

if "replay_progress" not in st.session_state:
    st.session_state["replay_progress"] = 0


def go_to_diagnose(batch_id):
    st.session_state["selected_batch"] = batch_id
    st.session_state["page"] = "이탈진단"


def go_to_forecast(batch_id):
    st.session_state["selected_batch"] = batch_id
    st.session_state["page"] = "농도예측"

def start_replay():
    st.session_state["replay_status"] = "재생 중"


def pause_replay():
    st.session_state["replay_status"] = "일시정지"


def reset_replay():
    st.session_state["replay_status"] = "대기 중"
    st.session_state["replay_progress"] = 0

# =================================================
# 사이드바
# =================================================
page = st.sidebar.radio(
    "메뉴",
    [
        "공정 모니터링",
        "이탈진단",
        "농도예측",
    ],
    key="page",
)


# =================================================
# Demo Reactor
# 실제 이탈진단 연결 후 status는 분석 결과로 교체
# =================================================
reactors = [
    {
        "reactor": "Reactor 1",
        "batch": 19,
        "strategy": "RC",
        "status": "정상",
    },
    {
        "reactor": "Reactor 2",
        "batch": 40,
        "strategy": "OC",
        "status": "정상",
    },
    {
        "reactor": "Reactor 3",
        "batch": 74,
        "strategy": "APC",
        "status": "주의",
    },
    {
        "reactor": "Reactor 4",
        "batch": 91,
        "strategy": "Fault(평가대상)",
        "status": "경고",
    },
]


# =================================================
# Demo Replay 시점
# Replay 100% = 선택된 각 Demo Batch의 공정 80% cut 시점
# =================================================
DEMO_CUT_TIME = {
    19: 165.6,
    40: 184.0,
    74: 184.0,
    91: 206.4,
}

replay_fraction = st.session_state["replay_progress"] / 100.0
process_progress = st.session_state["replay_progress"] * 0.8

replay_states = {}

for reactor in reactors:
    batch_id = reactor["batch"]

    batch_df = (
        df[df["배치번호"] == batch_id]
        .sort_values("발효시간")
        .copy()
    )

    if batch_df.empty:
        replay_states[batch_id] = None
        continue

    min_time = float(batch_df["발효시간"].min())

    target_time = DEMO_CUT_TIME[batch_id] * replay_fraction
    target_time = max(min_time, target_time)

    replay_states[batch_id] = get_replay_state(
        df=df,
        batch_id=batch_id,
        current_time=target_time,
    )


# =================================================
# UI Helper
# =================================================
def get_selected_reactor(batch_id):
    return next(
        reactor
        for reactor in reactors
        if reactor["batch"] == batch_id
    )


def get_comparison_strategy(batch_id, selected_state, selected_reactor):
    if selected_state is not None:
        strategy = selected_state.get("comparison_strategy")
        if strategy is not None:
            return strategy

    if batch_id == 91:
        return "APC"

    return selected_reactor["strategy"]


def status_class(status):
    if status == "정상":
        return "status-normal"
    if status == "주의":
        return "status-caution"
    return "status-warning"


def status_icon(status):
    if status == "정상":
        return "🟢"
    if status == "주의":
        return "🟡"
    return "🔴"


def render_status_pill(status):
    st.markdown(
        (
            f'<span class="status-pill {status_class(status)}">'
            f'{status_icon(status)} {status}</span>'
        ),
        unsafe_allow_html=True,
    )


def render_info_chips(items, status=None):
    chip_html = ""

    for label, value in items:
        chip_html += (
            '<div class="chip">'
            f'<span class="chip-label">{label}</span>'
            f'<strong>{value}</strong>'
            '</div>'
        )

    if status is not None:
        chip_html += (
            f'<span class="status-pill {status_class(status)}">'
            f'{status_icon(status)} {status}</span>'
        )

    st.markdown(
        f'<div class="chip-wrap">{chip_html}</div>',
        unsafe_allow_html=True,
    )


# =================================================
# PAGE 1 - 공정 모니터링
# =================================================
if page == "공정 모니터링":

    # ---------------------------------------------
    # 제목 + Replay 버튼
    # ---------------------------------------------
    title_col, start_col, pause_col, reset_col = st.columns(
        [5.4, 1.0, 1.2, 1.0]
    )

    with title_col:
        st.header("공정 모니터링")
        st.caption(
            f"Historical Batch Replay · "
            f"{st.session_state['replay_status']} · "
            f"Replay {st.session_state['replay_progress']}%"
        )

    with start_col:
        st.button(
            "▶ Replay",
            key="replay_start",
            on_click=start_replay,
            use_container_width=True,
        )

    with pause_col:
        st.button(
            "⏸ 일시정지",
            key="replay_pause",
            on_click=pause_replay,
            use_container_width=True,
        )

    with reset_col:
        st.button(
            "↺ 초기화",
            key="replay_reset",
            on_click=reset_replay,
            use_container_width=True,
        )

    # ---------------------------------------------
    # 상태 요약
    # ---------------------------------------------
    normal_count = sum(r["status"] == "정상" for r in reactors)
    caution_count = sum(r["status"] == "주의" for r in reactors)
    warning_count = sum(r["status"] == "경고" for r in reactors)

    sum1, sum2, sum3, sum4 = st.columns(4)

    with sum1:
        st.metric("현재 가동 Reactor", len(reactors))

    with sum2:
        st.metric("🟢 정상", normal_count)

    with sum3:
        st.metric("🟡 주의", caution_count)

    with sum4:
        st.metric("🔴 경고", warning_count)

    # ---------------------------------------------
    # 우선 확인 공정
    # ---------------------------------------------
    st.subheader("우선 확인 공정")

    status_priority = {
        "경고": 0,
        "주의": 1,
        "정상": 2,
    }

    priority_reactors = sorted(
        [
            reactor
            for reactor in reactors
            if reactor["status"] in ["경고", "주의"]
        ],
        key=lambda reactor: status_priority[reactor["status"]],
    )

    priority_cols = st.columns(len(priority_reactors))

    for col, reactor in zip(priority_cols, priority_reactors):
        with col:
            with st.container(border=True):
                head_col, button_col = st.columns([3.2, 1.0])

                with head_col:
                    st.markdown(
                        (
                            f'<div class="priority-title">'
                            f'{status_icon(reactor["status"])} '
                            f'{reactor["status"]} · '
                            f'{reactor["reactor"]} · '
                            f'Batch {reactor["batch"]}'
                            f'</div>'
                        ),
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        (
                            f'<div class="priority-body">'
                            f'{reactor["strategy"]} · '
                            f'공정 진행률 {process_progress:.1f}%<br>'
                            f'이탈점수·우선 확인 변수는 분석 모듈 연결 후 표시'
                            f'</div>'
                        ),
                        unsafe_allow_html=True,
                    )

                with button_col:
                    st.button(
                        "상세보기",
                        key=f"detail_{reactor['batch']}",
                        on_click=go_to_diagnose,
                        args=(reactor["batch"],),
                        use_container_width=True,
                    )

    # ---------------------------------------------
    # 현재 가동 공정 - 카드가 아닌 리스트
    # ---------------------------------------------
    st.subheader("현재 가동 공정")

    header_cols = st.columns([1.35, 1.10, 1.55, 1.30, 1.05, 1.10])

    headers = [
        "Reactor",
        "Batch",
        "전략/구분",
        "공정 진행률",
        "상태",
        "Action",
    ]

    for col, header in zip(header_cols, headers):
        with col:
            st.caption(f"**{header}**")

    for reactor in reactors:
        row_cols = st.columns([1.35, 1.10, 1.55, 1.30, 1.05, 1.10])

        with row_cols[0]:
            st.write(reactor["reactor"])

        with row_cols[1]:
            st.write(f"Batch {reactor['batch']}")

        with row_cols[2]:
            st.write(reactor["strategy"])

        with row_cols[3]:
            st.write(f"{process_progress:.1f}%")

        with row_cols[4]:
            render_status_pill(reactor["status"])

        with row_cols[5]:
            st.button(
                "상세보기",
                key=f"reactor_detail_{reactor['batch']}",
                on_click=go_to_diagnose,
                args=(reactor["batch"],),
                use_container_width=True,
            )

    st.caption(
        "※ 상태는 UI 개발용 임시값이며 실제 이탈진단 결과 연결 후 교체됩니다."
    )

    # ---------------------------------------------
    # Replay 자동 진행
    # ---------------------------------------------
    if st.session_state["replay_status"] == "재생 중":
        if st.session_state["replay_progress"] < 100:
            time.sleep(0.5)
            st.session_state["replay_progress"] += 1
            st.rerun()
        else:
            st.session_state["replay_status"] = "완료"
            st.rerun()


# =================================================
# PAGE 2 - 이탈진단
# =================================================
elif page == "이탈진단":

    from deviation_diagnosis import get_deviation_result, get_deviation_history
    from golden_profile import build_all_golden_profiles
    from config import STRATEGY_MEAN_DURATION
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import datetime as _dt

    # 그래프 한글 폰트 설정 (안 하면 축/범례 한글이 깨진 사각형으로 표시됨)
    _installed_fonts = {f.name for f in fm.fontManager.ttflist}
    for _font_candidate in ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        if _font_candidate in _installed_fonts:
            plt.rcParams["font.family"] = _font_candidate
            break
    plt.rcParams["axes.unicode_minus"] = False

    # 검증된 최종 성능(경고단독 기준, 이탈점수_분석_최종.ipynb) - 신규 분석 아님, 참고용 고정값
    VALIDATED_PERFORMANCE = {
        "재현율": 62.71, "FPR(오탐률)": 10.00, "정밀도": 20.85,
        "배치단위오탐률": 21.05, "에피소드탐지율": "17/21 (81.0%)",
    }

    # 기여 변수 유형별 판단 가이드 (도메인 지식 기반 참고표)
    # 주요 진단 카드에는 짧은 버전(SHORT), "판단 가이드" 참고표에는 자세한 버전(DETAIL)을 씀
    # - 4가지 색은 전부 "존재감 있는" 색으로 통일 (회색 계열은 안 씀 - 회색은
    #   비활성/덜 중요해 보인다는 피드백 반영. 1위든 3위든 유형색은 동일하고,
    #   순위는 카드의 숫자 배지·테두리 굵기로만 구분)
    TYPE_COLORS = {
        "제어반응변수": "#378ADD",
        "공정상태변수": "#B7791F",
        "강한 고정 제어변수": "#8B5CF6",
        "사전 계획 스케줄 변수": "#0D9488",
    }

    SHORT_GUIDE = {
        "제어반응변수": "제어루프·펌프·밸브 점검",
        "공정상태변수": "원인 제어변수 확인, 수율 영향 기록",
        "강한 고정 제어변수": "제어기·센서 고장 우선 점검",
        "사전 계획 스케줄 변수": "레시피·밸브·센서 확인",
    }

    DETAIL_GUIDE = {
        "제어반응변수": (
            "실시간 피드백으로 조절되는 변수입니다. 목표 설정값(SP)과 실제 투입량(PV)의 "
            "차이를 먼저 확인하고, 제어루프 튜닝 상태·펌프 유량·밸브 개도·도징 시스템 "
            "동작 여부를 순서대로 점검하세요. 일시적 노이즈인지 장비 이상인지 구분하는 "
            "것이 우선입니다."
        ),
        "공정상태변수": (
            "제어변수가 아니라 공정이 이미 반응한 '결과'라 지금 값을 되돌릴 수는 없습니다. "
            "대신 (1) 최종 수율에 영향이 있을지 기록하고, (2) 이 결과를 유발했을 가능성이 "
            "있는 제어반응변수(염기·산·당·냉난방수유량)의 최근 이력을 함께 확인해 원인을 "
            "역추적하세요."
        ),
        "강한 고정 제어변수": (
            "정상 운전에서는 거의 흔들리지 않도록 강하게 제어되는 변수입니다. 이 변수가 "
            "상위로 올라왔다면 단순 공정 변동이 아니라 제어기·센서 자체의 고장일 가능성이 "
            "있으므로, 다른 변수보다 먼저 센서 교정 상태와 제어 루프 동작 여부를 "
            "점검하세요."
        ),
        "사전 계획 스케줄 변수": (
            "정해진 레시피 스케줄대로 투입되어야 하는 변수입니다. 이탈이 나타났다면 "
            "레시피 설정이 변경되었는지, 또는 밸브·유량계·센서가 스케줄대로 동작하지 "
            "않고 있는지(고착·지연·고장)를 확인하세요."
        ),
    }

    VARIABLE_TYPE_GUIDE = {
        "염기투입유량_Fb": "제어반응변수",
        "산투입유량_Fa": "제어반응변수",
        "당공급유량_Fs": "제어반응변수",
        "냉난방수유량_Fc": "제어반응변수",
        "페니실린농도_P": "공정상태변수",
        "용존산소_DO2": "공정상태변수",
        "산소소비율_OUR": "공정상태변수",
        "pH": "강한 고정 제어변수",
        "발효온도_T": "강한 고정 제어변수",
        "공기주입유량_Fg": "사전 계획 스케줄 변수",
        "PAA투입유량_Fpaa": "사전 계획 스케줄 변수",
    }

    @st.cache_resource
    def get_golden_profiles_cached():
        return build_all_golden_profiles(df)

    golden_profiles = get_golden_profiles_cached()

    selected_batch = st.session_state["selected_batch"]
    selected_state = replay_states.get(selected_batch)
    selected_reactor = get_selected_reactor(selected_batch)

    # ---------------------------------------------
    # 실제 이탈진단 결과 연결 (get_deviation_result / get_deviation_history)
    # 확정 로직(11개 변수 SMAPE + 전략별 2.5σ + 1지점 주의/2지점 이상 경고)은
    # deviation_diagnosis.py 안에서 그대로 재사용되며 여기서는 결과만 받아 표시함
    # ---------------------------------------------
    diagnosis = None
    history = None
    if selected_state is not None:
        diagnosis = get_deviation_result(
            df=df,
            golden_profiles=golden_profiles,
            batch_id=selected_batch,
            current_time=selected_state["current_time"],
        )
        history = get_deviation_history(
            df=df,
            golden_profiles=golden_profiles,
            batch_id=selected_batch,
            current_time=selected_state["current_time"],
        )

    comparison_strategy = (
        diagnosis["comparison_strategy"]
        if diagnosis is not None
        else get_comparison_strategy(
            selected_batch,
            selected_state,
            selected_reactor,
        )
    )

    status = diagnosis["status"] if diagnosis is not None else selected_reactor["status"]

    remaining_hours = None
    if diagnosis is not None and comparison_strategy in STRATEGY_MEAN_DURATION:
        remaining_hours = max(
            0.0, STRATEGY_MEAN_DURATION[comparison_strategy] - diagnosis["current_time"]
        )

    # ---------------------------------------------
    # 경보 이력 로그 (세션 내 기록/추적용) — batch/시점 조합당 1회만 기록
    # ---------------------------------------------
    if "diagnosis_log" not in st.session_state:
        st.session_state["diagnosis_log"] = []

    past_logs_for_batch = [
        row for row in st.session_state["diagnosis_log"]
        if row["batch_id"] == selected_batch
    ]

    # "정상" 상태는 확인/조치가 필요한 경보가 아니므로 이력에 남기지 않음
    # (경보 이력 표에 정상 시점이 섞여 보이는 문제 수정)
    current_log_row = None
    if diagnosis is not None and diagnosis["status"] != "정상":
        log_key = (diagnosis["batch_id"], round(diagnosis["current_time"], 2))
        for row in st.session_state["diagnosis_log"]:
            if (row["batch_id"], round(row["current_time_h"], 2)) == log_key:
                current_log_row = row
                break
        if current_log_row is None:
            current_log_row = {
                "batch_id": diagnosis["batch_id"],
                "current_time_h": round(diagnosis["current_time"], 1),
                "progress_pct": diagnosis["progress"],
                "status": diagnosis["status"],
                "deviation_score": round(diagnosis["deviation_score"], 4),
                "threshold": round(diagnosis["threshold"], 4),
                "comparison_strategy": diagnosis["comparison_strategy"],
                "top3": ", ".join(f"{n}({s:.2f})" for n, s in diagnosis["top3"]),
                "note": "",
                "담당자": "",
                "기록시각": "",
            }
            st.session_state["diagnosis_log"].append(current_log_row)
            past_logs_for_batch = [
                row for row in st.session_state["diagnosis_log"]
                if row["batch_id"] == selected_batch
            ]

    # ---------------------------------------------
    # 제목 + Batch 정보칩 (breadcrumb 포함)
    # ---------------------------------------------
    st.caption("공정 모니터링 > " + f"Batch {selected_batch}")

    head_left, head_right = st.columns([1.15, 2.85])

    with head_left:
        st.header(f"Batch {selected_batch} 상세진단")
        st.caption("Golden 기준 비교 및 이탈진단")

    with head_right:
        chip_items = [
            ("배치 구분", selected_reactor["strategy"]),
            ("Golden 비교", f"{comparison_strategy} · n=6"),
            ("공정 진행률", f"{process_progress:.1f}%"),
        ]
        if remaining_hours is not None:
            chip_items.append(("잔여 공정시간", f"{remaining_hours:.1f}h"))
        render_info_chips(chip_items, status=status)

    # ---------------------------------------------
    # 핵심 진단 요약(종합 SMAPE / 최초 신호 / 지속 이탈 / 우선 확인 변수)
    # 커스텀 HTML로 렌더링 - 경고/주의일 때 점수 빨간 글씨 + Top3 박스 빨간 테두리
    # ---------------------------------------------
    if diagnosis is not None:
        is_alert = status != "정상"
        score_color = "#D1242F" if is_alert else "#111827"
        top3_border = "1px solid #FFB3B3" if is_alert else "1px solid #D9DEE8"
        top3_bg = "#FFF5F5" if is_alert else "#FFFFFF"

        first_signal_txt = (
            f"{history['first_signal_time']:.1f}h"
            if history and history["first_signal_time"] is not None
            else "없음"
        )
        persist_txt = f"{history['persist_hours']:.1f}h" if history else "-"
        top3_names = " · ".join(name for name, _ in diagnosis["top3"])

        st.markdown(
            f"""
            <div style="display:flex; gap:0.6rem; align-items:stretch; margin:0.25rem 0 0.4rem 0;">
              <div style="flex:1; border:1px solid #D9DEE8; border-radius:10px; padding:0.5rem 0.8rem;">
                <div style="font-size:0.74rem; color:#6B7280;">종합 SMAPE 점수 (임계값)</div>
                <div style="font-size:1.7rem; font-weight:800; color:{score_color}; line-height:1.2;">
                    {diagnosis['deviation_score']:.2f}
                    <span style="font-size:0.85rem; font-weight:500; color:#6B7280;">/ {diagnosis['threshold']:.2f}</span>
                </div>
              </div>
              <div style="flex:1; border:1px solid #D9DEE8; border-radius:10px; padding:0.5rem 0.8rem;">
                <div style="font-size:0.74rem; color:#6B7280;">최초 신호</div>
                <div style="font-size:1.45rem; font-weight:700;">{first_signal_txt}</div>
              </div>
              <div style="flex:1; border:1px solid #D9DEE8; border-radius:10px; padding:0.5rem 0.8rem;">
                <div style="font-size:0.74rem; color:#6B7280;">지속 이탈</div>
                <div style="font-size:1.45rem; font-weight:700;">{persist_txt}</div>
              </div>
              <div style="flex:1.4; border:{top3_border}; background:{top3_bg}; border-radius:10px; padding:0.5rem 0.8rem;">
                <div style="font-size:0.74rem; color:#6B7280;">우선 확인 변수</div>
                <div style="font-size:1.0rem; font-weight:700; color:#D1242F;">{top3_names}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.container(border=True):
            diag1, diag2, diag3, diag4 = st.columns([1.0, 1.0, 1.0, 1.4])
            with diag1:
                st.caption("종합 SMAPE 점수")
                st.write("## 연결 전")
            with diag2:
                st.caption("최초 신호")
                st.write("## 연결 전")
            with diag3:
                st.caption("지속 이탈")
                st.write("## 연결 전")
            with diag4:
                st.caption("우선 확인 변수")
                st.write("### 연결 전")

    # ---------------------------------------------
    # Golden 비교 그래프(탭) + 주요 진단
    # ---------------------------------------------
    graph_col, diagnosis_col = st.columns([2.3, 1.1])

    deviation_variables = [
        "페니실린농도_P",
        "용존산소_DO2",
        "pH",
        "발효온도_T",
        "염기투입유량_Fb",
        "냉난방수유량_Fc",
        "산소소비율_OUR",
        "공기주입유량_Fg",
        "당공급유량_Fs",
        "산투입유량_Fa",
        "PAA투입유량_Fpaa",
    ]

    # 주요 진단 카드의 "그래프 보기"를 누르면 아래를 그 변수 하나만 크게 보여주는
    # 화면으로 전환 (st.tabs는 코드로 활성 탭을 바꿀 방법이 없어서, 평소엔 원래대로
    # 탭 11개를 그대로 쓰고 - 이게 더 낫다는 피드백 반영 - 점프했을 때만 탭 대신
    # 단일 그래프 + "탭으로 돌아가기" 버튼을 보여주는 방식으로 처리)
    if "page2_jump_var" not in st.session_state:
        st.session_state["page2_jump_var"] = None

    def _focus_var(v):
        st.session_state["page2_jump_var"] = v

    def _clear_focus():
        st.session_state["page2_jump_var"] = None

    def _render_var_plot(var):
        if history is None:
            st.info(
                "그래프 연결 전 · Golden 기준선과 현재 Batch 궤적을 "
                "현재 시점까지만 표시"
            )
            return
        var_hist = history["variables"][var]
        t = np.array(var_hist["time"])
        actual = np.array(var_hist["actual"])
        gmean = np.array(var_hist["golden_mean"])

        # 밴드(±σ) 시각화는 사용하지 않음 - 확정 방법론은 골든배치 "평균선" 대비
        # SMAPE 거리 하나로만 이탈을 계산하며, 밴드 기반 판정을 쓰지 않기로 확정함
        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        ax.plot(t, gmean, color="#378ADD", linewidth=1.5, label="골든배치 평균")
        ax.plot(
            t, actual, color="#D1242F", linewidth=1.8,
            label=f"현재 Batch {selected_batch}",
        )

        if history["first_signal_time"] is not None:
            ax.axvline(
                history["first_signal_time"], color="#D1242F",
                linestyle="--", linewidth=1.2, label="최초 신호",
            )
        ax.axvline(t[-1], color="#6B7280", linestyle=":", linewidth=1.2, label="현재 시점")

        ax.set_xlabel("시간 (h)", fontsize=10, labelpad=4)
        ax.set_ylabel(var, fontsize=10, labelpad=4)
        ax.margins(x=0.01)
        ax.legend(fontsize=9, loc="upper left", ncol=4, frameon=False)
        ax.tick_params(labelsize=9, pad=2)
        fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.11)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with graph_col:
        st.subheader("현재 Batch vs 골든배치 기준 프로파일")

        jump_var = st.session_state.get("page2_jump_var")
        if jump_var in deviation_variables:
            st.button("← 전체 변수 탭으로 돌아가기", on_click=_clear_focus, key="back_to_tabs")
            st.caption(f"주요 진단에서 선택한 변수: **{jump_var}**")
            _render_var_plot(jump_var)
        else:
            var_tabs = st.tabs(deviation_variables)
            for tab, var in zip(var_tabs, deviation_variables):
                with tab:
                    _render_var_plot(var)

    with diagnosis_col:
        st.subheader("주요 진단")

        if diagnosis is not None and history is not None:
            if status == "정상":
                st.markdown(
                    "<div style='border:1px solid #B7E4C7; background:#E9F8EE; "
                    "border-radius:8px; padding:0.6rem 0.8rem; font-size:0.85rem;'>"
                    "🟢 골든배치 기준 대비 정상 범위 내에서 운전 중입니다.</div>",
                    unsafe_allow_html=True,
                )
            else:
                # 카드 스타일은 순위와 무관하게 전부 동일(테두리 굵기·배경 톤 고정) —
                # 굵기를 순위별로 다르게 했더니 "왜 칸마다 폭이 다르냐"는 혼란을 줘서
                # 되돌림. 순위 표시는 숫자 배지 하나로만 하고, 색은 오직 "변수 유형"만
                # 나타낸다 (파랑=제어반응변수·갈색=공정상태변수·보라=강한 고정 제어변수·
                # 청록=사전 계획 스케줄 변수, 회색 없음).
                for rank, (name, score) in enumerate(diagnosis["top3"], start=1):
                    var_hist = history["variables"][name]
                    rising = var_hist["actual"][-1] > var_hist["golden_mean"][-1]
                    direction = "지속 상승 중" if rising else "기준보다 낮게 지속 이탈"
                    var_type = VARIABLE_TYPE_GUIDE.get(name, "기타")
                    guide_color = TYPE_COLORS.get(var_type, "#374151")
                    guide_text = SHORT_GUIDE.get(var_type, "")
                    score_txt = f"{score:.2f}" if not np.isnan(score) else "—"
                    repeat_count = sum(
                        1 for row in past_logs_for_batch if name in row["top3"]
                    )
                    repeat_txt = (
                        f" · 반복 {repeat_count}/{len(past_logs_for_batch)}"
                        if past_logs_for_batch
                        else ""
                    )

                    card_col, jump_col = st.columns([5.8, 0.7])
                    with card_col:
                        card_html = f"""
                        <div style="border-left:4px solid {guide_color}; background:#FAFAFA;
                                    border-radius:6px; padding:0.5rem 0.7rem; margin-bottom:0.5rem;">
                          <div style="display:flex; justify-content:space-between; align-items:baseline;">
                            <span style="font-weight:700; font-size:0.92rem;">
                              <span style="display:inline-block; width:1.1rem; height:1.1rem; line-height:1.1rem;
                                           text-align:center; border-radius:50%; background:{guide_color};
                                           color:#fff; font-size:0.68rem; margin-right:0.3rem;">{rank}</span>{name}
                            </span>
                            <span style="font-weight:800; font-size:0.98rem; color:#D1242F;">{score_txt}</span>
                          </div>
                          <div style="font-size:0.74rem; color:#374151; margin-top:0.15rem;">
                            {direction}{repeat_txt}
                            <span style="font-size:0.66rem; color:{guide_color}; border:1px solid {guide_color};
                                         border-radius:6px; padding:0 0.3rem; margin-left:0.2rem;
                                         white-space:nowrap; display:inline-block;">{var_type}</span>
                          </div>
                          <div style="font-size:0.72rem; color:#6B7280; margin-top:0.1rem;">💡 {guide_text}</div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                    with jump_col:
                        # 카드 자체는 순수 HTML이라 클릭시 Streamlit과 상호작용할 수 없어서,
                        # 옆에 아이콘만 작게 "이 변수 그래프 보기"를 제공 (테두리 없는
                        # tertiary 타입 - 버튼처럼 안 보이고 아이콘만 있는 느낌으로)
                        st.button(
                            "📈",
                            key=f"jump_{name}_{selected_batch}_{round(diagnosis['current_time'], 2)}",
                            on_click=_focus_var,
                            args=(name,),
                            help=f"{name} 그래프에서 보기",
                            type="tertiary",
                        )
        else:
            st.write("실제 이탈진단 연결 후 표시")

        if diagnosis is not None and current_log_row is not None and status != "정상":
            note_key = f"note_{selected_batch}_{round(diagnosis['current_time'], 2)}"

            prev_note = current_log_row.get("note", "")

            # 이 <style> 자체도 화면에 보이는 하나의 st 엘리먼트라서, 컬럼 안쪽에
            # 넣으면 그 컬럼에만 위아래 여백이 하나 더 생겨서 "담당자"/"조치 메모"
            # 줄이 서로 어긋나 보였음(직전 스크린샷 문제) — 컬럼을 만들기 전에
            # 미리 한 번만 주입해서 두 칸의 시작 위치가 정확히 맞도록 수정.
            st.markdown(
                """
                <style>
                .st-key-page2_note_box textarea {
                    min-height: 2.55rem !important;
                    height: 2.55rem !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            person_col, note_col = st.columns([1.0, 2.2])
            with person_col:
                st.caption("담당자")
                person = st.text_input(
                    "담당자",
                    key=f"person_{selected_batch}_{round(diagnosis['current_time'], 2)}",
                    value=current_log_row.get("담당자", ""),
                    placeholder="이름",
                    label_visibility="collapsed",
                )
            with note_col:
                # st.text_input(단일 줄)은 브라우저가 저장된 다른 폼 데이터로 자동완성
                # 제안을 띄우는 문제가 있었어서(예: "병원 진료") text_area로 유지하고,
                # 위에서 주입한 CSS로 담당자 칸과 같은 높이로 강제 조정.
                st.caption("조치 메모")
                with st.container(key="page2_note_box"):
                    note = st.text_area(
                        "조치 메모",
                        key=note_key,
                        value=prev_note,
                        placeholder="예: 밸브 점검 요청함",
                        label_visibility="collapsed",
                    )

            current_log_row["note"] = note
            current_log_row["담당자"] = person

            # 메모 내용이 실제로 바뀐 시점에만 기록시각을 새로 찍음
            # (페이지가 다시 그려질 때마다 시각이 계속 "지금"으로 갱신되는 것을 방지)
            if note != prev_note:
                current_log_row["기록시각"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

            if current_log_row.get("기록시각"):
                st.caption(
                    f"마지막 기록: {current_log_row.get('담당자') or '담당자 미기재'} · "
                    f"{current_log_row['기록시각']}"
                )

    # ---------------------------------------------
    # 판단 가이드 (기여 변수 유형별 대응 방법 참고표 + 과거 검증 성능 + 추세)
    # ---------------------------------------------
    with st.expander("판단 가이드", expanded=False):
        if diagnosis is not None and history is not None:
            st.write("**추세**")
            curve = history["deviation_curve"]
            if len(curve) >= 6 and not (np.isnan(curve[-1]) or np.isnan(curve[-6])):
                recent, prior = curve[-1], curve[-6]
                if recent > prior * 1.05:
                    st.write("📈 최근 구간에서 이탈점수 **상승 추세**")
                elif recent < prior * 0.95:
                    st.write("📉 최근 구간에서 이탈점수 **하락(회복) 추세**")
                else:
                    st.write("➡️ 최근 구간 큰 변화 없음")
            else:
                st.write("추세 판단에 필요한 구간이 아직 부족합니다.")

            st.write("**이 경보 유형의 과거 검증 성능**")
            st.caption(
                f"경고 기준 정밀도 {VALIDATED_PERFORMANCE['정밀도']}% · "
                f"재현율 {VALIDATED_PERFORMANCE['재현율']}% · "
                f"배치단위 오탐률 {VALIDATED_PERFORMANCE['배치단위오탐률']}%"
                " — 경고 5건 중 약 1건 정도가 실제 고장과 일치. 단독 신호만으로 "
                "확신하지 말고 기여도·추세·유형별 가이드와 함께 판단."
            )

        st.write("**기여 변수 유형별 판단 가이드**")
        guide_rows = "".join(
            f"<tr>"
            f"<td style='padding:0.3rem 0.5rem; border-bottom:1px solid #EEE; "
            f"color:{TYPE_COLORS[vtype]}; font-weight:600; white-space:nowrap;'>{vtype}</td>"
            f"<td style='padding:0.3rem 0.5rem; border-bottom:1px solid #EEE; line-height:1.5;'>{DETAIL_GUIDE[vtype]}</td>"
            f"</tr>"
            for vtype in TYPE_COLORS
        )
        st.markdown(
            f"<table style='width:100%; border-collapse:collapse; font-size:0.8rem;'>"
            f"<tr><th style='text-align:left; padding:0.3rem 0.5rem; border-bottom:2px solid #D9DEE8;'>기여 변수 유형</th>"
            f"<th style='text-align:left; padding:0.3rem 0.5rem; border-bottom:2px solid #D9DEE8;'>판단 가이드</th></tr>"
            f"{guide_rows}</table>",
            unsafe_allow_html=True,
        )

    # ---------------------------------------------
    # 변수별 상세 데이터 보기 (11개 변수 전체, 유형별 색상 표시)
    # ---------------------------------------------
    with st.expander("변수별 상세 데이터 보기", expanded=False):
        if history is not None:
            top3_names_set = {name for name, _ in diagnosis["top3"]} if diagnosis else set()

            rows_html = ""
            for var in deviation_variables:
                var_hist = history["variables"][var]
                actual_last = var_hist["actual"][-1]
                gmean_last = var_hist["golden_mean"][-1]
                denom = abs(actual_last) + abs(gmean_last)
                var_score = abs(actual_last - gmean_last) / denom if denom > 1e-6 else float("nan")
                score_display = f"{var_score:.3f}" if not np.isnan(var_score) else "—"
                var_type = VARIABLE_TYPE_GUIDE.get(var, "기타")
                type_color = TYPE_COLORS.get(var_type, "#374151")
                is_top3 = var in top3_names_set
                row_bg = "background:#FFF5F5;" if is_top3 else ""
                mark = "⚠ " if is_top3 else ""

                rows_html += (
                    f"<tr style='{row_bg}'>"
                    f"<td style='padding:0.3rem 0.5rem; border-bottom:1px solid #EEE;'>{mark}{var}</td>"
                    f"<td style='padding:0.3rem 0.5rem; border-bottom:1px solid #EEE; color:{type_color}; font-weight:600;'>{var_type}</td>"
                    f"<td style='padding:0.3rem 0.5rem; border-bottom:1px solid #EEE;'>{actual_last:.3f}</td>"
                    f"<td style='padding:0.3rem 0.5rem; border-bottom:1px solid #EEE;'>{gmean_last:.3f}</td>"
                    f"<td style='padding:0.3rem 0.5rem; border-bottom:1px solid #EEE;'>{score_display}</td>"
                    f"</tr>"
                )

            st.markdown(
                f"<table style='width:100%; border-collapse:collapse; font-size:0.8rem;'>"
                f"<tr>"
                f"<th style='text-align:left; padding:0.3rem 0.5rem; border-bottom:2px solid #D9DEE8;'>변수</th>"
                f"<th style='text-align:left; padding:0.3rem 0.5rem; border-bottom:2px solid #D9DEE8;'>변수 유형</th>"
                f"<th style='text-align:left; padding:0.3rem 0.5rem; border-bottom:2px solid #D9DEE8;'>현재값</th>"
                f"<th style='text-align:left; padding:0.3rem 0.5rem; border-bottom:2px solid #D9DEE8;'>골든 기준(평균)</th>"
                f"<th style='text-align:left; padding:0.3rem 0.5rem; border-bottom:2px solid #D9DEE8;'>변수별 이탈점수(SMAPE)</th>"
                f"</tr>{rows_html}</table>",
                unsafe_allow_html=True,
            )
            st.caption(
                "빨간 배경(⚠)은 현재 Top3 기여 변수 · '—'는 실측값·골든기준이 "
                "모두 0에 가까워 이탈점수를 정의할 수 없는 구간입니다. "
                "유형별 대응 방법은 위 '판단 가이드'를 참고하세요."
            )
        else:
            st.write("연결 전")

    # ---------------------------------------------
    # 기록/추적 - 배치별 경보 이력 + 리포트 다운로드
    # ---------------------------------------------
    with st.expander("경보 이력 / 리포트", expanded=False):
        if past_logs_for_batch:
            log_df = pd.DataFrame(past_logs_for_batch).rename(columns={
                "batch_id": "배치",
                "current_time_h": "시간(h)",
                "progress_pct": "진행률(%)",
                "status": "상태",
                "deviation_score": "이탈점수",
                "threshold": "임계값",
                "comparison_strategy": "Golden비교",
                "top3": "우선확인변수",
                "note": "조치메모",
            })
            st.dataframe(log_df, use_container_width=True, hide_index=True)

            csv_bytes = log_df.to_csv(index=False).encode("utf-8-sig")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    f"Batch {selected_batch} 이력 CSV",
                    data=csv_bytes,
                    file_name=f"batch{selected_batch}_diagnosis_log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                if diagnosis is not None:
                    top3_lines = "\n".join(
                        f"- {n} ({VARIABLE_TYPE_GUIDE.get(n, '기타')}): "
                        f"이탈점수 {s:.3f} · {SHORT_GUIDE.get(VARIABLE_TYPE_GUIDE.get(n, ''), '')}"
                        for n, s in diagnosis["top3"]
                    )
                    report_text = (
                        f"# Batch {selected_batch} 이탈진단 리포트\n\n"
                        f"- 조회 시점: {diagnosis['current_time']:.1f}h "
                        f"(진행률 {diagnosis['progress']:.1f}%)\n"
                        f"- 배치 구분: {selected_reactor['strategy']} / "
                        f"Golden 비교: {comparison_strategy}\n"
                        f"- 상태: {status}\n"
                        f"- 종합 SMAPE 점수: {diagnosis['deviation_score']:.3f} "
                        f"(임계값 {diagnosis['threshold']:.3f})\n"
                        f"- 최초 신호: {first_signal_txt} · 지속 이탈: {persist_txt}\n"
                        f"- 담당자: {(current_log_row or {}).get('담당자') or '미기재'}"
                        f"{' (기록시각: ' + current_log_row['기록시각'] + ')' if current_log_row and current_log_row.get('기록시각') else ''}\n"
                        f"- 조치 메모: {(current_log_row or {}).get('note') or '(없음)'}\n\n"
                        f"## 우선 확인 변수\n{top3_lines}\n"
                    )
                    st.download_button(
                        "배치 리포트 (.md)",
                        data=report_text.encode("utf-8"),
                        file_name=f"batch{selected_batch}_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
        else:
            st.write("이 배치에서 아직 기록된 진단 이력이 없습니다.")

        all_log_df = pd.DataFrame(st.session_state["diagnosis_log"])
        if not all_log_df.empty:
            all_csv_bytes = all_log_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "전체 배치 이력 CSV 다운로드",
                data=all_csv_bytes,
                file_name="all_diagnosis_log.csv",
                mime="text/csv",
            )
        st.caption(
            "이력은 이번 브라우저 세션 동안 조회한 시점만 누적됩니다 "
            "(서버 재시작/새 세션 시 초기화)."
        )

    move_left, move_right = st.columns([4.2, 1.0])

    with move_right:
        st.button(
            "12시간 후 농도예측 →",
            on_click=go_to_forecast,
            args=(selected_batch,),
            use_container_width=True,
        )


# =================================================
# PAGE 3 - 농도예측
# =================================================
elif page == "농도예측":

    selected_batch = st.session_state["selected_batch"]
    selected_state = replay_states.get(selected_batch)
    selected_reactor = get_selected_reactor(selected_batch)

    comparison_strategy = get_comparison_strategy(
        selected_batch,
        selected_state,
        selected_reactor,
    )

    if selected_state is not None:
        current_time = selected_state["current_time"]
        current_p = selected_state["current_p"]
        prediction_time = current_time + 12.0
    else:
        current_time = None
        current_p = None
        prediction_time = None

    # ---------------------------------------------
    # 제목 + Batch 정보칩
    # ---------------------------------------------
    head_left, head_right = st.columns([1.35, 2.65])

    with head_left:
        st.header(f"Batch {selected_batch} · 12시간 농도예측")
        st.caption("현재 시점까지의 공정정보 기반 예측")

    with head_right:
        render_info_chips(
            [
                ("전략/구분", selected_reactor["strategy"]),
                ("Golden 비교", comparison_strategy),
                ("공정 진행률", f"{process_progress:.1f}%"),
                ("Horizon", "12h"),
            ]
        )

    # ---------------------------------------------
    # 핵심 KPI - 현재P / 12h예측P / 변화량
    # ---------------------------------------------
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        with st.container(border=True):
            if current_p is not None:
                st.metric(
                    "현재 페니실린 농도 P(t)",
                    f"{current_p:.3f} g/L",
                )
            else:
                st.metric("현재 페니실린 농도 P(t)", "-")

    with kpi2:
        with st.container(border=True):
            st.metric(
                "12시간 후 예측 농도 P(t+12h)",
                "연결 전",
            )

    with kpi3:
        with st.container(border=True):
            st.metric(
                "예상 변화량 ΔP",
                "연결 전",
            )

    if current_time is not None:
        st.caption(
            f"현재 시점 {current_time:.1f} h · "
            f"예측 시점 {prediction_time:.1f} h (+12h)"
        )
    else:
        st.caption("현재 시점 - · 예측 시점 -")

    # ---------------------------------------------
    # 예측 그래프 + 예측 해석
    # ---------------------------------------------
    graph_col, side_col = st.columns([2.55, 1.0])

    with graph_col:
        st.subheader("농도 추이 및 12시간 후 예측")

        with st.container(border=True):
            st.info(
                "그래프 연결 전 · 현재 시점까지 실제 농도 궤적과 "
                "12시간 후 예측값 1개 지점을 표시"
            )

            st.caption(
                "표시 예정: 실제 관측 농도 · 현재 시점 · "
                "12시간 후 예측 지점"
            )

    with side_col:
        st.subheader("예측 해석")

        with st.container(border=True):
            st.write("**현재 결과**")
            st.write("12시간 후 예측모델 연결 전")

            st.divider()

            st.write("**확인 사항**")
            st.write(
                "이탈 신호가 관찰된 구간에서는 "
                "예측값과 현재 공정 상태를 함께 확인"
            )

        with st.expander("모델 정보", expanded=False):
            st.write("모델 · **LightGBM**")
            st.write("Horizon · **12h**")
            st.write("평가 지표 · **최종 결과 연결 예정**")

    st.caption(
        "※ 예측값은 현재 시점까지 관측된 정보만 사용하며, "
        "Replay의 미래 실제값은 검증용으로만 비교합니다."
    )

    back_col, empty_col = st.columns([1.25, 4.0])

    with back_col:
        st.button(
            "← 상세진단으로",
            on_click=go_to_diagnose,
            args=(selected_batch,),
            use_container_width=True,
        )
