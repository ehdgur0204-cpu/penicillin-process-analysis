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

    selected_batch = st.session_state["selected_batch"]
    selected_state = replay_states.get(selected_batch)
    selected_reactor = get_selected_reactor(selected_batch)

    comparison_strategy = get_comparison_strategy(
        selected_batch,
        selected_state,
        selected_reactor,
    )

    status = selected_reactor["status"]

    # ---------------------------------------------
    # 제목 + Batch 정보칩
    # ---------------------------------------------
    head_left, head_right = st.columns([1.15, 2.85])

    with head_left:
        st.header(f"Batch {selected_batch} 상세진단")
        st.caption("Golden 기준 비교 및 이탈진단")

    with head_right:
        render_info_chips(
            [
                ("배치 구분", selected_reactor["strategy"]),
                ("Golden 비교", comparison_strategy),
                ("공정 진행률", f"{process_progress:.1f}%"),
            ],
            status=status,
        )

    st.caption(
        "※ 현재 상태는 UI 임시값 · 실제 이탈진단 모듈 연결 후 교체"
    )

    # ---------------------------------------------
    # 핵심 진단 요약 - Top3까지 한 줄
    # ---------------------------------------------
    with st.container(border=True):
        diag1, diag2, diag3, diag4 = st.columns([1.0, 1.0, 1.0, 1.55])

        with diag1:
            st.caption("이탈점수")
            st.write("## 연결 전")

        with diag2:
            st.caption("최초 이탈 신호")
            st.write("## 연결 전")

        with diag3:
            st.caption("연속 임계값 초과")
            st.write("## 연결 전")

        with diag4:
            st.caption("우선 확인 변수 Top 3")
            st.write("### 연결 전 · 연결 전 · 연결 전")

        st.caption(
            "판정: 전략별 2.5σ 임계값 · "
            "1개 시점 초과=주의 · 2개 시점 이상 연속 초과=경고"
        )

    # ---------------------------------------------
    # Golden 비교 그래프 + 주요 진단
    # ---------------------------------------------
    graph_col, diagnosis_col = st.columns([2.55, 1.0])

    with graph_col:
        st.subheader("Golden 기준선 비교")

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

        selected_variable = st.selectbox(
            "확인할 공정변수",
            deviation_variables,
            label_visibility="collapsed",
        )

        with st.container(border=True):
            st.write(f"**{selected_variable}**")

            st.info(
                "그래프 연결 전 · Golden 기준선과 현재 Batch 궤적을 "
                "현재 시점까지만 표시"
            )

            st.caption(
                "표시 예정: Golden 기준선 · 현재 Batch 궤적 · "
                "최초 이탈 신호 · 현재 시점"
            )

    with diagnosis_col:
        st.subheader("주요 진단")

        with st.container(border=True):
            st.write("**현재 관측 정보**")

            if selected_state is not None:
                st.write(
                    f"발효시간 · **{selected_state['current_time']:.1f} h**"
                )
                st.write(
                    f"페니실린 농도 · "
                    f"**{selected_state['current_p']:.3f} g/L**"
                )
            else:
                st.write("-")

            st.divider()

            st.write("**진단 해석**")
            st.write("실제 이탈진단 연결 후 표시")

            st.caption(
                "연관 패턴과 우선 점검 변수 확인을 위한 "
                "의사결정 보조 정보"
            )

    # ---------------------------------------------
    # 상세 변수 영역은 기본 화면에서 접어둠
    # ---------------------------------------------
    with st.expander("변수별 상세 상태 보기", expanded=False):
        variable_header = st.columns([2.2, 1.2, 1.2, 1.2])

        for col, label in zip(
            variable_header,
            ["변수", "현재값", "Golden 기준", "변수별 이탈점수"],
        ):
            with col:
                st.caption(f"**{label}**")

        example_cols = st.columns([2.2, 1.2, 1.2, 1.2])

        with example_cols[0]:
            st.write(selected_variable)

        with example_cols[1]:
            if (
                selected_state is not None
                and selected_variable == "페니실린농도_P"
            ):
                st.write(f"{selected_state['current_p']:.3f}")
            else:
                st.write("연결 전")

        with example_cols[2]:
            st.write("연결 전")

        with example_cols[3]:
            st.write("연결 전")

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
