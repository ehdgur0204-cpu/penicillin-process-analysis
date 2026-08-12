import streamlit as st
import time
import sys
import functools
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


# 프로젝트 최상위 폴더
ROOT_DIR = Path(__file__).resolve().parent.parent

# 프로젝트 루트와 src를 Python 경로에 추가
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from predict_12h import predict_12h
from data_loader import load_data
from replay_utils import get_replay_state
import feature_engineering as fe


# =================================================
# [신규 기능용] 배포 번들(model + golden_band_artifact) 직접 로드
# -- predict_12h.py 내부의 private 헬퍼(_load_bundle 등)에 의존하지 않고
#    이 파일에서 독립적으로 로드한다. predict_12h.py 버전이 바뀌어도
#    (private 함수명이 다르거나 없어도) 안전하게 동작하도록 하기 위함.
#    predict_12h() 자체는 원래 쓰시던 그대로, 공개 함수만 그대로 사용.
# =================================================
_BUNDLE_CANDIDATES = [
    Path("final_model_bundle.joblib"),                     # streamlit 실행 cwd 기준
    ROOT_DIR / "final_model_bundle.joblib",                 # 프로젝트 루트
    ROOT_DIR / "src" / "final_model_bundle.joblib",         # src 폴더
    ROOT_DIR / "streamlit" / "final_model_bundle.joblib",   # streamlit 폴더
]


def _resolve_bundle_path():
    for p in _BUNDLE_CANDIDATES:
        if p.exists():
            return p
    # 못 찾으면 첫 후보 그대로 반환 -> joblib.load에서 명확한 FileNotFoundError 발생시킴
    return _BUNDLE_CANDIDATES[0]


@functools.lru_cache(maxsize=1)
def _load_bundle():
    return joblib.load(_resolve_bundle_path())


def _slice_upto_now(df, batch_id, current_time):
    """predict_12h.py와 동일 로직(사본): 현재 시점까지로 잘라 정렬."""
    sub = df[(df[fe.BATCH_COL] == batch_id) & (df[fe.TIME_COL] <= current_time)]
    sub = sub.sort_values(fe.TIME_COL).reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"Batch {batch_id}: current_time={current_time} 이하 데이터가 없습니다.")
    return sub


def _build_feature_row(sub, batch_id, artifact):
    """predict_12h.py와 동일 로직(사본): 24개 feature 1행 생성."""
    now_row = sub.iloc[[-1]].copy()
    feat = {c: now_row[c].to_numpy()[0] for c in fe.BASE21}
    feat["GB_score_cum_v2"] = fe.gb_score_cum_v2_at_current(artifact, sub, batch_id)
    feat["OUR_slope20h"] = fe.slope_at_current(sub, "Oxygen Uptake Rate(OUR:(g min^{-1}))")
    feat["S_slope20h"] = fe.slope_at_current(sub, "Substrate concentration(S:g/L)")
    row = pd.DataFrame([feat])[fe.FINAL_FEATURES]
    return row.rename(columns=fe.SAFE_COLMAP)


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

    /* ---- 농도예측 페이지: 아이콘 배지 KPI 카드 ---- */
    .kpi-card {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        height: 100%;
    }

    .kpi-icon {
        flex-shrink: 0;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 19px;
    }

    .kpi-icon-current   { background: #E7F1FF; }
    .kpi-icon-predicted { background: #F1EAFE; }
    .kpi-icon-up        { background: #E9F8EE; }
    .kpi-icon-down      { background: #FFF0F0; }
    .kpi-icon-flat      { background: #F1F2F4; }

    .kpi-text { display: flex; flex-direction: column; gap: 0.05rem; }
    .kpi-label { font-size: 0.78rem; color: #6B7280; font-weight: 600; }
    .kpi-value { font-size: 1.30rem; font-weight: 800; color: #1F2937; line-height: 1.2; }
    .kpi-sub { font-size: 0.72rem; color: #9CA3AF; }

    /* ---- 농도예측 페이지: 비교 정보 표 ---- */
    .compare-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
    }

    .compare-table th, .compare-table td {
        padding: 0.42rem 0.20rem;
        border-bottom: 1px solid #EEF0F2;
        text-align: left;
    }

    .compare-table th {
        color: #6B7280;
        font-weight: 600;
        width: 55%;
    }

    .compare-table td {
        color: #1F2937;
        font-weight: 700;
    }

    .compare-table tr:last-child th,
    .compare-table tr:last-child td {
        border-bottom: none;
    }

    /* ---- Demo Mode 토글 라벨 ---- */
    .demo-mode-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #6B7280;
        text-align: right;
        margin-bottom: 0.10rem;
    }

    /* ---- 상태 판정 배너 ---- */
    .status-banner {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        border-radius: 10px;
        padding: 0.55rem 0.85rem;
        margin: 0.35rem 0 0.55rem 0;
        font-size: 0.86rem;
    }

    .banner-ok      { background: #E9F8EE; border: 1px solid #B7E4C7; }
    .banner-caution { background: #FFF9E6; border: 1px solid #F5D77A; }
    .banner-danger  { background: #FFF0F0; border: 1px solid #FFB3B3; }

    .banner-icon { font-size: 1.35rem; line-height: 1; }
    .banner-title { font-weight: 800; color: #1F2937; }
    .banner-body { font-size: 0.78rem; color: #4B5563; }
    .banner-tag {
        margin-left: auto;
        font-size: 0.68rem;
        color: #9CA3AF;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 0.08rem 0.4rem;
        white-space: nowrap;
    }

    /* ---- 영향 변수 리스트 ---- */
    .factor-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.80rem;
        padding: 0.28rem 0;
        border-bottom: 1px solid #F3F4F6;
    }
    .factor-row:last-child { border-bottom: none; }
    .factor-rank { color: #9CA3AF; width: 1.4rem; flex-shrink: 0; }
    .factor-name { color: #1F2937; flex-grow: 1; }
    .factor-up   { color: #D1242F; font-weight: 700; }
    .factor-down { color: #2563EB; font-weight: 700; }
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

@st.cache_data
def get_prediction_data():
    return pd.read_csv(ROOT_DIR / "data" / "raw" / "merged_data.csv")

prediction_df = get_prediction_data()



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

if "demo_mode" not in st.session_state:
    # 리플레이(과거 배치) 데이터이므로 기본값 ON.
    # 실전 운영 데이터 연결 시에는 반드시 False로 시작해야 함 (미래값 노출 방지).
    st.session_state["demo_mode"] = True

if "replay_history" not in st.session_state:
    # 페이지3에서 리플레이 시점을 옮길 때마다 (배치, 시점)별 예측/실제값을 누적 저장
    # -> "리플레이 세션 누적 검증" 섹션의 러닝 R²/MAE/산점도에 사용
    st.session_state["replay_history"] = {}


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


def render_kpi_card(icon_class, icon, label, value, sub=None):
    # 반드시 한 줄짜리 문자열로 만든다 -- 여러 줄 들여쓰기된 f-string을 그대로 st.markdown에
    # 넘기면, sub가 없을 때 생기는 빈 줄 때문에 마크다운 파서가 HTML 블록을 중간에 끊어버려서
    # 닫는 </div> 태그가 파싱되지 않고 화면에 그대로(리터럴 텍스트로) 노출되는 버그가 있었음.
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    html = (
        f'<div class="kpi-card">'
        f'<div class="kpi-icon {icon_class}">{icon}</div>'
        f'<div class="kpi-text">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def get_actual_at_time(prediction_df, batch_id, target_time, tol=0.05):
    """merged_data.csv 원본 컬럼(Batch_ID / Time (h) / Penicillin concentration(P:g/L))
    기준으로 특정 시점의 '실제' 농도를 찾는다. Demo Mode 전용 -- 실전 운영에서는
    아직 관측되지 않은 미래 시점이므로 호출하면 안 됨 (리플레이 검증 용도 전용)."""
    cand = prediction_df[prediction_df["Batch_ID"] == batch_id].copy()
    if cand.empty:
        return None
    cand["_dist"] = (cand["Time (h)"] - target_time).abs()
    cand = cand.sort_values("_dist")
    nearest = cand.iloc[0]
    if nearest["_dist"] > tol and nearest["_dist"] > 1.0:
        return None
    return float(nearest["Penicillin concentration(P:g/L)"])


# =================================================
# [신규] 예측구간 / 상태판정 기준값
# -- analyze_for_dashboard_features.py 에서 GroupKFold(5) OOF 잔차와
#    정상배치 90개의 실측 12h ΔP 분포로부터 산출한 값. 하드코딩된 임의값이 아님.
#    (재현: analyze_for_dashboard_features.py 참고)
# =================================================

# GroupKFold(5) OOF 잔차(실제-예측)의 10th/90th percentile -> 80% 경험적 예측구간 폭
PI_LOW_OFFSET = -1.79
PI_HIGH_OFFSET = 1.77

# 정상배치(1~90) 12h 실측 ΔP 분포의 10th/90th, 1st/99th percentile
# 주의: 이 데이터셋에서는 Fault 배치가 '더 큰' ΔP가 아니라 오히려 정체/약한 상승으로
# 나타나는 경우가 많아(생산 정체), 아래 기준은 상승/하강 양방향을 함께 봄.
# ※ 실제 검증된 이탈 판정 지표는 '이탈진단' 페이지의 GB_score_cum_v2이며,
#    아래 기준은 어디까지나 Demo용 보조 표시임 (모델 출력이 아님).
DELTA_NORMAL_LOW, DELTA_NORMAL_HIGH = -0.4, 2.5
DELTA_EXTREME_LOW, DELTA_EXTREME_HIGH = -1.8, 2.9


def classify_delta(delta_p):
    """(status, direction) 반환. status: 정상/주의/이탈위험, direction: 'up'/'down'/None"""
    if delta_p is None:
        return None, None
    if delta_p > DELTA_EXTREME_HIGH:
        return "이탈위험", "up"
    if delta_p < DELTA_EXTREME_LOW:
        return "이탈위험", "down"
    if delta_p > DELTA_NORMAL_HIGH:
        return "주의", "up"
    if delta_p < DELTA_NORMAL_LOW:
        return "주의", "down"
    return "정상", None


def render_status_banner(delta_p):
    status, direction = classify_delta(delta_p)
    if status is None:
        return
    banner_cls = {"정상": "banner-ok", "주의": "banner-caution", "이탈위험": "banner-danger"}[status]
    icon = {"정상": "🟢", "주의": "🟡", "이탈위험": "🔴"}[status]
    if direction == "up":
        headline = "12시간 후 농도 상승 위험" if status == "이탈위험" else "12시간 후 농도 상승 주의"
    elif direction == "down":
        headline = "12시간 후 농도 정체·감소 위험" if status == "이탈위험" else "12시간 후 농도 정체·감소 주의"
    else:
        headline = "12시간 후 정상 범위 내 변화 예상"
    body = f"현재 대비 {delta_p:+.3f} g/L 변화 예상 (정상배치 90개 실측 분포 기준 {DELTA_NORMAL_LOW}~{DELTA_NORMAL_HIGH} g/L 벗어남)" \
        if status != "정상" else f"현재 대비 {delta_p:+.3f} g/L 변화 예상 — 정상배치 분포 범위 내"
    st.markdown(
        f"""
        <div class="status-banner {banner_cls}">
            <div class="banner-icon">{icon}</div>
            <div>
                <div class="banner-title">{headline}</div>
                <div class="banner-body">{body}</div>
            </div>
            <div class="banner-tag">Demo 기준 · 참고용</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =================================================
# [신규] SHAP 기반 개별 예측 영향 변수 (TreeExplainer, 캐시)
# =================================================
@st.cache_resource
def get_shap_explainer():
    if not _SHAP_AVAILABLE:
        return None
    bundle = _load_bundle()
    return shap.TreeExplainer(bundle["model"])


_INV_SAFE_COLMAP = {v: k for k, v in fe.SAFE_COLMAP.items()}

# 화면에 보여줄 한글 라벨 (안전 컬럼명 -> 짧은 한글 표기). 매핑에 없는 변수는 원본 영문명을 그대로 씀.
FEATURE_LABEL_KO = {
    "Time (h)": "발효 경과시간",
    "Substrate concentration(S:g/L)": "기질농도(S)",
    "S_slope20h": "기질농도 20h 기울기",
    "OUR_slope20h": "OUR 20h 기울기",
    "GB_score_cum_v2": "Golden 대비 누적 이탈점수",
    "Dissolved oxygen concentration(DO2:mg/L)": "용존산소(DO2)",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))": "산소소비율(OUR)",
    "Carbon evolution rate(CER:g/h)": "이산화탄소 발생률(CER)",
    "carbon dioxide percent in off-gas(CO2outgas:%)": "배가스 CO2 비율",
    "PAA flow(Fpaa:PAA flow (L/h))": "PAA 투입유량",
    "Generated heat(Q:kJ)": "발생 열량(Q)",
    "Sugar feed rate(Fs:L/h)": "당 공급유량(Fs)",
    "Aeration rate(Fg:L/h)": "공기주입유량(Fg)",
    "pH(pH:pH)": "pH",
    "Temperature(T:K)": "발효온도(T)",
}


def get_top_shap_factors(batch_id, current_time, top_n=5):
    """선택된 배치·시점의 예측에 대해 SHAP 기여도 상위 top_n개 변수를 반환.
    [{'name':.., 'shap':.., 'direction':'up'/'down'}, ...] 또는 SHAP 미설치 시 None."""
    explainer = get_shap_explainer()
    if explainer is None:
        return None
    bundle = _load_bundle()
    artifact = bundle["golden_band_artifact"]
    sub = _slice_upto_now(prediction_df, batch_id, current_time)
    X_now = _build_feature_row(sub, batch_id, artifact)
    sv = np.array(explainer.shap_values(X_now)).flatten()
    order = np.argsort(-np.abs(sv))[:top_n]
    factors = []
    for i in order:
        col_safe = X_now.columns[i]
        col_real = _INV_SAFE_COLMAP.get(col_safe, col_safe)
        label = FEATURE_LABEL_KO.get(col_real, col_real)
        factors.append({
            "name": label,
            "shap": float(sv[i]),
            "direction": "up" if sv[i] > 0 else "down",
        })
    return factors


def render_shap_factors(factors):
    if factors is None:
        st.info("SHAP 라이브러리가 설치되어 있지 않아 영향 변수를 계산할 수 없습니다. (`pip install shap` 필요)")
        return
    rows_html = ""
    for i, f in enumerate(factors, start=1):
        arrow = "▲" if f["direction"] == "up" else "▼"
        arrow_cls = "factor-up" if f["direction"] == "up" else "factor-down"
        rows_html += (
            '<div class="factor-row">'
            f'<span class="factor-rank">{i}</span>'
            f'<span class="factor-name">{f["name"]}</span>'
            f'<span class="{arrow_cls}">{arrow} 영향</span>'
            '</div>'
        )
    st.markdown(rows_html, unsafe_allow_html=True)
    st.caption("SHAP(TreeExplainer) 기준, 이 배치·이 시점 예측에 대한 개별 기여도 상위 5개 (전역 중요도 아님)")


# =================================================
# [신규] Golden Batch 비교 곡선 조회
# =================================================
def get_golden_curve(strategy):
    """배포 모델에 저장된 golden band artifact에서 전략별 평균 농도 곡선을
    (시간축 배열, 농도 배열)로 반환. predict_12h/이탈진단 페이지와 동일한 artifact 사용."""
    bundle = _load_bundle()
    artifact = bundle["golden_band_artifact"]
    if strategy not in artifact["full_bands"]:
        return None, None
    band = artifact["full_bands"][strategy][fe.TARGET_COL]
    dur = artifact["strategy_mean_duration"][strategy]
    time_grid = artifact["prog_grid"] * dur
    return time_grid, band


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
# PAGE 3 - 농도예측  (리디자인 v2 -- 전체 기능 반영)
# =================================================
elif page == "농도예측":

    # ---------------------------------------------
    # [신규] 배치 선택 + 리플레이 시점 슬라이더
    # ---------------------------------------------
    BATCH_OPTIONS = [(19, "RC · Batch 19"), (40, "OC · Batch 40"),
                      (74, "APC · Batch 74"), (91, "Fault · Batch 91")]
    BATCH_LABELS = dict(BATCH_OPTIONS)
    BATCH_IDS = [b for b, _ in BATCH_OPTIONS]
    # 리플레이 세션 누적 검증 산점도에서 전략별 색상 구분용
    STRATEGY_OF_BATCH = {19: "RC", 40: "OC", 74: "APC", 91: "Fault"}
    STRATEGY_COLOR = {"RC": "#3FA796", "OC": "#0B5E59", "APC": "#4A3AA7", "Fault": "#D9704F"}

    def _on_batch_change():
        st.session_state["selected_batch"] = st.session_state["page3_batch_choice"]

    selected_batch = st.session_state["selected_batch"]
    if selected_batch not in BATCH_IDS:
        selected_batch = BATCH_IDS[0]
        st.session_state["selected_batch"] = selected_batch

    selected_reactor = get_selected_reactor(selected_batch)

    ctrl_col1, ctrl_col2 = st.columns([1.1, 3.0])

    with ctrl_col1:
        st.selectbox(
            "배치 선택",
            options=BATCH_IDS,
            format_func=lambda b: BATCH_LABELS[b],
            index=BATCH_IDS.index(selected_batch),
            key="page3_batch_choice",
            on_change=_on_batch_change,
        )

    batch_df = df[df["배치번호"] == selected_batch].sort_values("발효시간")
    batch_min_time = float(batch_df["발효시간"].min())
    batch_max_time = float(batch_df["발효시간"].max())
    demo_mode_pre = st.session_state.get("demo_mode", True)
    slider_hi = max(batch_min_time, batch_max_time - 12.0) if demo_mode_pre else batch_max_time
    # 기본값은 전역 replay_progress(0일 수 있음)를 그대로 물려받지 않고, 이 배치의
    # 대표(80% 진행률) 컷 시점으로 잡는다. replay_progress=0일 때 그대로 물려받으면
    # current_time이 배치 시작 직후로 잡혀 예측값이 0에 가깝게 나오는 문제(이전에 보고된
    # localhost:8502 화면의 근사-0 값 증상)가 재현되기 때문.
    default_time = DEMO_CUT_TIME.get(selected_batch, batch_min_time)
    default_time = float(np.clip(default_time, batch_min_time, slider_hi))

    with ctrl_col2:
        current_time = st.slider(
            "리플레이 시점 조정 (h)",
            min_value=round(batch_min_time, 1),
            max_value=round(slider_hi, 1),
            value=round(default_time, 1),
            step=1.0,
            key=f"page3_time_slider_{selected_batch}",
        )

    selected_state = get_replay_state(df=df, batch_id=selected_batch, current_time=current_time)

    comparison_strategy = get_comparison_strategy(
        selected_batch,
        selected_state,
        selected_reactor,
    )

    if selected_state is not None:
        current_time = selected_state["current_time"]
        current_p = selected_state["current_p"]
        prediction_time = current_time + 12.0

        prediction_result = predict_12h(
            df=prediction_df,
            batch_id=selected_batch,
            current_time=current_time,
        )

        predicted_p = prediction_result["predicted_p_12h"]
        delta_p = prediction_result["change"]

    else:
        current_time = None
        current_p = None
        prediction_time = None
        predicted_p = None
        delta_p = None

    # Demo Mode 전용: 리플레이 데이터이므로 예측 시점의 '실제값'을 미리 알 수 있음.
    demo_mode = st.session_state.get("demo_mode", True)
    actual_p_future = None
    if demo_mode and selected_state is not None:
        actual_p_future = get_actual_at_time(prediction_df, selected_batch, prediction_time)

    # ---------------------------------------------
    # 제목 + Batch 정보칩 + Demo Mode 토글
    # ---------------------------------------------

    head_left, head_mid, head_right = st.columns([1.30, 2.15, 1.05])

    with head_left:
        st.header(f"Batch {selected_batch} · 12시간 농도예측")
        st.caption("현재 시점까지의 공정정보 기반 예측")

    with head_mid:
        render_info_chips(
            [
                ("전략/구분", selected_reactor["strategy"]),
                ("Golden 비교", comparison_strategy),
                ("공정 진행률", f"{process_progress:.1f}%"),
                ("Horizon", "12h"),
            ]
        )

    with head_right:
        st.markdown(
            '<div class="demo-mode-label">Demo Mode</div>',
            unsafe_allow_html=True,
        )
        st.toggle(
            "Demo Mode ON" if demo_mode else "Demo Mode OFF",
            value=demo_mode,
            key="demo_mode",
            label_visibility="collapsed",
        )

    # ---------------------------------------------
    # 핵심 KPI - 아이콘 배지 카드 (현재P / 12h예측P(+구간) / 변화량)
    # ---------------------------------------------

    kpi1, kpi2, kpi3 = st.columns(3)

    # 세 카드의 세로 높이를 동일하게 고정 (현재 농도 카드는 sub 텍스트가 없어
    # 다른 두 카드보다 짧아 보이는 문제가 있었음 -- 셋 다 같은 높이로 통일)
    KPI_CARD_HEIGHT = 118

    with kpi1:
        with st.container(border=True, height=KPI_CARD_HEIGHT):
            value = f"{current_p:.3f} g/L" if current_p is not None else "-"
            sub = f"발효 경과시간 {current_time:.1f} h 기준" if current_time is not None else None
            render_kpi_card(
                "kpi-icon-current", "🧪",
                "현재 페니실린 농도 P(t)", value, sub=sub,
            )

    with kpi2:
        with st.container(border=True, height=KPI_CARD_HEIGHT):
            if predicted_p is not None:
                value = f"{predicted_p:.3f} g/L"
                pi_lo = predicted_p + PI_LOW_OFFSET
                pi_hi = predicted_p + PI_HIGH_OFFSET
                sub = f"80% 확률 예측 범위 {pi_lo:.2f}~{pi_hi:.2f} g/L"
            else:
                value, sub = "-", None
            render_kpi_card(
                "kpi-icon-predicted", "⏱",
                "12시간 후 예측 농도 P(t+12h)", value, sub=sub,
            )

    with kpi3:
        with st.container(border=True, height=KPI_CARD_HEIGHT):
            if delta_p is not None:
                if delta_p > 0.001:
                    icon_class, icon, value = "kpi-icon-up", "▲", f"+{delta_p:.3f} g/L"
                elif delta_p < -0.001:
                    icon_class, icon, value = "kpi-icon-down", "▼", f"{delta_p:.3f} g/L"
                else:
                    icon_class, icon, value = "kpi-icon-flat", "→", "변화 없음"
            else:
                icon_class, icon, value = "kpi-icon-flat", "→", "-"
            render_kpi_card(icon_class, icon, "예상 변화량 ΔP", value,
                             sub="GroupKFold(5) OOF R²=0.9760" if delta_p is not None else None)

    if current_time is not None:
        st.caption(
            f"현재 시점 {current_time:.1f} h · "
            f"예측 시점 {prediction_time:.1f} h (+12h)"
        )
    else:
        st.caption("현재 시점 - · 예측 시점 -")

    # ---------------------------------------------
    # 예측 그래프 + 예측 해석 / 비교 정보 / 영향 변수 / 모델 정보
    # ---------------------------------------------

    graph_col, side_col = st.columns([2.55, 1.0])

    with graph_col:

        st.subheader("농도 추이 및 12시간 후 예측")

        with st.container(border=True):

            # ---------------------------------------------
            # 실제 관측 데이터
            # ---------------------------------------------
            batch_data = df[
                (df["배치번호"] == selected_batch)
                & (df["발효시간"] <= current_time)
            ].copy()

            batch_data = batch_data[
                ["발효시간", "페니실린농도_P"]
            ]

            batch_data = batch_data.rename(
                columns={
                    "발효시간": "Time",
                    "페니실린농도_P": "Penicillin",
                }
            )

            # ---------------------------------------------
            # Plotly 그래프 생성
            # ---------------------------------------------
            fig = go.Figure()

            # ⓪ [신규] Golden Batch 비교 곡선 (배경 참조선)
            golden_time, golden_curve = get_golden_curve(comparison_strategy)
            if golden_time is not None:
                fig.add_trace(
                    go.Scatter(
                        x=golden_time,
                        y=golden_curve,
                        mode="lines",
                        name=f"Golden Batch ({comparison_strategy})",
                        line=dict(width=2, color="#9CA3AF", dash="dot"),
                        hovertemplate=(
                            "Golden 기준선<br>시간: %{x:.1f} h<br>농도: %{y:.3f} g/L"
                            "<extra></extra>"
                        ),
                    )
                )

            # ① 실제 농도 추이 → 실선
            if not batch_data.empty:

                fig.add_trace(
                    go.Scatter(
                        x=batch_data["Time"],
                        y=batch_data["Penicillin"],
                        mode="lines",
                        name="실제 농도",
                        line=dict(
                            width=3,
                            color="#2B6CB0",
                        ),
                        hovertemplate=(
                            "발효시간: %{x:.1f} h<br>"
                            "페니실린 농도: %{y:.3f} g/L"
                            "<extra></extra>"
                        ),
                    )
                )

            # ---------------------------------------------
            # ② 현재 시점 → 점으로 강조 + 세로 기준선
            # ---------------------------------------------
            if current_time is not None and current_p is not None:

                fig.add_trace(
                    go.Scatter(
                        x=[current_time],
                        y=[current_p],
                        mode="markers",
                        name="현재 시점",
                        marker=dict(
                            size=11,
                            symbol="circle",
                            color="#2B6CB0",
                        ),
                        hovertemplate=(
                            "현재 시점<br>"
                            "시간: %{x:.1f} h<br>"
                            "농도: %{y:.3f} g/L"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.add_vline(
                    x=current_time,
                    line_width=1.3,
                    line_dash="dash",
                    line_color="#9CA3AF",
                    annotation_text="현재 시점",
                    annotation_position="top",
                    annotation_font=dict(size=11, color="#6B7280"),
                )

            # ---------------------------------------------
            # ③ 현재 → 12시간 후 예측 → 점선 (+ 예측구간 밴드)
            # ---------------------------------------------
            if (
                current_time is not None
                and current_p is not None
                and prediction_time is not None
                and predicted_p is not None
            ):

                # [신규] 80% 확률 예측 범위 밴드 (예측점 주변 세로 막대 형태로 표시, 파스텔 연두)
                fig.add_trace(
                    go.Scatter(
                        x=[prediction_time, prediction_time],
                        y=[predicted_p + PI_LOW_OFFSET, predicted_p + PI_HIGH_OFFSET],
                        mode="lines",
                        name="80% 확률 예측 범위",
                        line=dict(width=6, color="rgba(163,230,53,0.35)"),
                        hoverinfo="skip",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=[current_time, prediction_time],
                        y=[current_p, predicted_p],
                        mode="lines",
                        name="12시간 후 예측 구간",
                        line=dict(
                            dash="dash",
                            width=2,
                            color="#16A34A",
                        ),
                        hovertemplate=(
                            "예측 구간<br>"
                            "시간: %{x:.1f} h<br>"
                            "농도: %{y:.3f} g/L"
                            "<extra></extra>"
                        ),
                    )
                )

                # ---------------------------------------------
                # ④ 12시간 후 예측값 → 초록색 동그라미
                # ---------------------------------------------
                fig.add_trace(
                    go.Scatter(
                        x=[prediction_time],
                        y=[predicted_p],
                        mode="markers+text",
                        name="12시간 후 예측",
                        marker=dict(
                            size=14,
                            symbol="circle",
                            color="#16A34A",
                            line=dict(color="white", width=1.2),
                        ),
                        text=[
                            f"예측<br>{predicted_p:.3f} g/L"
                        ],
                        textposition="top center",
                        hovertemplate=(
                            "12시간 후 예측<br>"
                            "시간: %{x:.1f} h<br>"
                            "예측 농도: %{y:.3f} g/L"
                            "<extra></extra>"
                        ),
                    )
                )

                # ---------------------------------------------
                # ⑤ [Demo Mode 전용] 예측 시점의 실제값 주석 → 노란 별
                # ---------------------------------------------
                if demo_mode and actual_p_future is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[prediction_time],
                            y=[actual_p_future],
                            mode="markers+text",
                            name="실제값 (Demo Mode 전용)",
                            marker=dict(
                                size=15,
                                symbol="star",
                                color="#FACC15",
                                line=dict(color="#92700E", width=1),
                            ),
                            text=[
                                f"실제값 {actual_p_future:.3f} g/L<br>(Demo Mode 전용)"
                            ],
                            textposition="bottom center",
                            textfont=dict(color="#92700E", size=11),
                            hovertemplate=(
                                "실제값 (Demo Mode 전용)<br>"
                                "시간: %{x:.1f} h<br>"
                                "실제 농도: %{y:.3f} g/L"
                                "<extra></extra>"
                            ),
                        )
                    )

                    # 리플레이 세션 히스토리에 누적 (배치+시점 단위로 dedup)
                    hist_key = f"{selected_batch}_{round(current_time, 1)}"
                    st.session_state["replay_history"][hist_key] = {
                        "batch": selected_batch,
                        "time": round(current_time, 1),
                        "predicted": predicted_p,
                        "actual": actual_p_future,
                        "strategy": STRATEGY_OF_BATCH.get(selected_batch, "기타"),
                    }

            # ---------------------------------------------
            # 그래프 레이아웃
            # ---------------------------------------------
            fig.update_layout(
                height=430,

                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20,
                ),

                plot_bgcolor="white",
                paper_bgcolor="white",

                xaxis=dict(
                    title="발효시간 (h)",
                    showgrid=True,
                    gridcolor="#EEF0F2",
                    zeroline=False,
                    showline=True,
                    linecolor="#D1D5DB",
                ),

                yaxis=dict(
                    title="페니실린 농도 (g/L)",
                    showgrid=True,
                    gridcolor="#EEF0F2",
                    zeroline=False,
                    showline=True,
                    linecolor="#D1D5DB",
                    rangemode="tozero",
                ),

                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    bgcolor="rgba(255,255,255,0)",
                ),

                hovermode="x unified",

                font=dict(
                    family="Arial, sans-serif",
                    size=13,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.caption(
            f"점선 회색: Golden Batch({comparison_strategy}) 평균 곡선 · "
            "현재 배치가 Golden 대비 어느 위치에 있는지, 12시간 후 그 차이가 어떻게 될지 함께 확인할 수 있습니다."
        )

        # ---------------------------------------------
        # [신규] 예측 영향 요인 -- 그래프 박스 바로 아래, 가로 폭을 위 박스에
        # 맞춰 전체 폭으로 배치 (비교 정보는 오른쪽 side_col로 이동)
        # ---------------------------------------------
        st.write("")
        st.subheader("예측 영향 요인")
        with st.container(border=True, height=190):
            if selected_state is not None:
                factors = get_top_shap_factors(selected_batch, current_time)
                render_shap_factors(factors)
            else:
                st.write("-")

    # ---------------------------------------------
    # 예측 해석 / 비교 정보 / 모델 정보
    # ---------------------------------------------

    with side_col:

        # [신규] 상태 판정 배너 -- "예측 해석" 박스 안이 아니라 그 위, 별도 요소로 배치.
        # 배치 선택창/헤더 바로 아래 딱 붙지 않도록 약간의 여백을 두고 시작.
        st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
        render_status_banner(delta_p)
        st.write("")

        st.subheader("예측 해석")

        with st.container(border=True, height=180):

            if predicted_p is not None:

                bullet_lines = [
                    f"12시간 후 페니실린 농도는 **{predicted_p:.3f} g/L**로 예측됩니다.",
                ]

                if delta_p > 0.001:
                    bullet_lines.append(
                        f"현재 대비 **+{delta_p:.3f} g/L 증가**할 것으로 예상됩니다."
                    )
                elif delta_p < -0.001:
                    bullet_lines.append(
                        f"현재 대비 **{delta_p:.3f} g/L 감소**할 것으로 예상됩니다."
                    )
                else:
                    bullet_lines.append(
                        "현재 농도와 유사한 수준으로 유지될 것으로 예상됩니다."
                    )

                pi_lo = predicted_p + PI_LOW_OFFSET
                pi_hi = predicted_p + PI_HIGH_OFFSET
                bullet_lines.append(
                    f"80% 확률로 **{pi_lo:.2f}~{pi_hi:.2f} g/L** 범위 안에 들 것으로 추정됩니다 "
                    "(GroupKFold(5) 잔차 분포 기반 경험적 구간)."
                )

                st.markdown("\n".join(f"- {line}" for line in bullet_lines))

            else:
                st.write(
                    "현재 시점의 공정 데이터가 없어 예측값을 표시할 수 없습니다."
                )

        # ---------------------------------------------
        # [신규] 비교 정보 -- 예측 해석 바로 아래로 이동
        # ---------------------------------------------
        st.write("")
        st.subheader("비교 정보")
        with st.container(border=True, height=190):
            rows = [
                ("현재 농도", f"{current_p:.3f} g/L" if current_p is not None else "-"),
                ("예측 농도 (t+12h)", f"{predicted_p:.3f} g/L" if predicted_p is not None else "-"),
                (
                    "실제값 (Demo Mode 전용)",
                    f"{actual_p_future:.3f} g/L"
                    if (demo_mode and actual_p_future is not None) else "-",
                ),
                (
                    "예측오차",
                    f"{abs(predicted_p - actual_p_future):.3f} g/L"
                    if (
                        demo_mode
                        and predicted_p is not None
                        and actual_p_future is not None
                    ) else "-",
                ),
            ]

            table_html = '<table class="compare-table">' + "".join(
                f"<tr><th>{label}</th><td>{value}</td></tr>"
                for label, value in rows
            ) + "</table>"

            st.markdown(table_html, unsafe_allow_html=True)

        st.write("")
        st.subheader("모델 정보")
        with st.expander("세부 수치는 접어서 확인", expanded=False):
            st.write("최종 모델 · **LightGBM Regressor** (24 features)")
            st.write("예측 Horizon · **12시간**")
            st.write("평가 지표 · **GroupKFold(5) OOF** 기준 R²=0.9760, MAE=1.0122 g/L")

        # ---------------------------------------------
        # [신규] 상세진단으로 이동 -- 모델 정보 바로 아래로 이동
        # ---------------------------------------------
        st.write("")
        st.button(
            "← 상세진단으로",
            key="back_to_diagnose_from_forecast",
            on_click=go_to_diagnose,
            args=(selected_batch,),
            use_container_width=True,
        )

    # ---------------------------------------------
    # 하단 안내
    # ---------------------------------------------

    st.caption(
        "※ 예측값은 현재 시점까지 관측된 정보만 사용하며, "
        "Demo Mode의 실제값·예측구간·상태판정은 리플레이 검증/시연용 참고 정보입니다."
    )
