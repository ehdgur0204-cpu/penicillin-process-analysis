# ============================================
# deviation_diagnosis.py
# 이탈진단 담당 모듈
#
# 기존에 확정한 이탈진단 로직(이탈점수_분석_최종.ipynb)을 바꾸지 않고
# 대시보드에서 호출 가능한 함수 형태로 정리했습니다.
#
# 확정 로직 (변경 없음):
#   - 이탈점수: SMAPE(|실제-골든평균| / (|실제|+|골든평균|)), 11개 변수 평균
#     (z-score/PCA 등 다른 정규화 추가 없음)
#   - 임계값: 전략별 Train 정상배치 SMAPE 곡선의 지점별 평균 + 2.5*표준편차
#     (SIGMA_K 재추정 없음)
#   - 3단계 판정: 지속조건(연속 지점 수)으로 주의/경고 구분
#       1개 지점만 임계값 초과 -> 주의
#       2개 지점 이상 연속 초과 -> 경고
#     (새로운 시그마 임계값을 추가한 게 아니라, 기존 2.5시그마 임계값 하나를
#      "얼마나 오래 넘었는지"로 나눈 것 — MIN_RUN=2)
#
# Golden Profile 12개 변수와 최종 SMAPE 11개 변수는 구분해서 사용합니다.
# (기질농도_S는 Golden Profile에는 있지만 최종 SMAPE 계산에는 넣지 않습니다)
#
# ⚠ 진행률(progress) 계산 관련 — 팀에 공유 필요한 부분
# replay_utils.calculate_progress()는 실제 strategy가 RC/OC/APC가 아닌 경우
# (Fault 배치) OVERALL_MEAN_DURATION(228.127h)을 사용합니다.
# 하지만 이탈진단 최종 검증에서는 Fault 배치도 비교전략(comparison_strategy,
# 예: APC)의 전략별 평균 운전시간(STRATEGY_MEAN_DURATION)을 사용했습니다
# (Fault91 -> APC 평균 225.458h). 이 모듈은 검증된 방식을 그대로 유지하기 위해
# replay_utils의 progress를 쓰지 않고 자체적으로 다시 계산합니다.
# ============================================

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d

from config import (
    BATCH_COL,
    STRATEGY_COL,
    TIME_COL,
    GOLDEN_BATCHES,
    TEST_NORMAL_BATCHES,
    TEST_FAULT_BATCHES,
    STRATEGY_MEAN_DURATION,
    FAULT_COMPARISON_STRATEGY,
    GOLDEN_STEP_COLS,
    SMAPE_COLS,
)

# --------------------------------------------
# 확정된 상수 (재추정하지 않음)
# --------------------------------------------
SIGMA_K = 2.5
MIN_RUN = 2  # 2지점 이상 연속 초과 = 경고
N_POINTS = 101
COMMON_POINTS = np.linspace(0, 1, N_POINTS)
_STEP_COLS = set(GOLDEN_STEP_COLS) & set(SMAPE_COLS)

# 임계값 캐시 (배치마다 새로 계산하지 않도록 모듈 전역에 보관)
_THRESHOLD_CACHE = {}


# --------------------------------------------
# 내부 유틸
# --------------------------------------------

def _get_comparison_strategy(batch_id, strategy):
    """이탈진단 시 사용할 Golden 비교전략. RC/OC/APC는 그대로, 나머지는 FAULT_COMPARISON_STRATEGY 조회."""
    if strategy in STRATEGY_MEAN_DURATION:
        return strategy
    return FAULT_COMPARISON_STRATEGY.get(batch_id)


def _get_train_normal_batches(df, strategy):
    """골든배치 + 고정 Test 배치를 제외한 나머지 같은 전략 배치 = Train 정상배치."""
    exclude = (
        set(GOLDEN_BATCHES.get(strategy, []))
        | set(TEST_NORMAL_BATCHES)
        | set(TEST_FAULT_BATCHES)
    )
    batch_ids = df.loc[df[STRATEGY_COL] == strategy, BATCH_COL].unique()
    return [int(b) for b in batch_ids if int(b) not in exclude]


def _interpolate_to_points(progress_known, values_known, points, column):
    """관측된 (progress, value)를 지정된 진행률 격자(points)에 보간.
    연속형: PCHIP / 계단형(step): 이전값 유지 — Golden Profile 생성 방식과 동일."""
    progress_known = np.asarray(progress_known, dtype=float)
    values_known = np.asarray(values_known, dtype=float)

    if column in _STEP_COLS:
        interpolator = interp1d(
            progress_known, values_known, kind="previous",
            bounds_error=False, fill_value=(values_known[0], values_known[-1]),
        )
        return interpolator(points)

    # 연속형: PCHIP은 중복 x값이 있으면 에러가 나므로 정리
    progress_unique, idx = np.unique(progress_known, return_index=True)
    values_unique = values_known[idx]
    if len(progress_unique) < 2:
        return np.full_like(points, values_unique[0] if len(values_unique) else np.nan, dtype=float)

    interpolator = PchipInterpolator(progress_unique, values_unique, extrapolate=False)
    result = interpolator(points)
    result = np.where(points < progress_unique[0], values_unique[0], result)
    result = np.where(points > progress_unique[-1], values_unique[-1], result)
    return result


def _smape_curve_for_batch(df, batch_id, mean_duration, golden_ref_means, points):
    """한 배치의 SMAPE_COLS 11개 변수 평균 이탈점수 곡선(points 길이)을 계산."""
    sub = df[df[BATCH_COL] == batch_id].sort_values(TIME_COL)
    if sub.empty:
        return np.full_like(points, np.nan, dtype=float)

    progress_known = np.minimum(sub[TIME_COL].to_numpy() / mean_duration, 1.0)

    scores = []
    for col in SMAPE_COLS:
        actual = _interpolate_to_points(progress_known, sub[col].to_numpy(), points, col)
        ref_mean = golden_ref_means[col]
        denom = np.abs(actual) + np.abs(ref_mean)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(denom < 1e-6, np.nan, np.abs(actual - ref_mean) / denom)
        scores.append(score)

    return np.nanmean(np.array(scores), axis=0)


def _filter_min_run(flag, min_run=MIN_RUN):
    """연속 min_run지점 이상인 구간만 True로 남기고, 나머지(1지점 blip)는 False."""
    flag = np.asarray(flag)
    result = np.zeros_like(flag, dtype=bool)
    i, n = 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j < n and flag[j]:
                j += 1
            if j - i >= min_run:
                result[i:j] = True
            i = j
        else:
            i += 1
    return result


def _build_thresholds(df, golden_profiles):
    """전략별 임계값 곡선(mean + 2.5*std, Train 정상배치 기준)을 계산 (최초 1회, 이후 캐시)."""
    thresholds = {}
    for strategy in ["RC", "OC", "APC"]:
        profile = golden_profiles[strategy]
        golden_ref_means = {col: profile[f"{col}_mean"].to_numpy() for col in SMAPE_COLS}
        mean_duration = STRATEGY_MEAN_DURATION[strategy]

        train_batches = _get_train_normal_batches(df, strategy)
        curves = np.array([
            _smape_curve_for_batch(df, b, mean_duration, golden_ref_means, COMMON_POINTS)
            for b in train_batches
        ])
        thresholds[strategy] = np.nanmean(curves, axis=0) + SIGMA_K * np.nanstd(curves, axis=0)

    return thresholds


def _get_thresholds(df, golden_profiles):
    cache_key = id(golden_profiles)
    if cache_key not in _THRESHOLD_CACHE:
        _THRESHOLD_CACHE[cache_key] = _build_thresholds(df, golden_profiles)
    return _THRESHOLD_CACHE[cache_key]


# --------------------------------------------
# 공개 함수
# --------------------------------------------

def get_deviation_result(df, golden_profiles, batch_id, current_time):
    """
    현재 Batch / 현재 시점의 이탈진단 결과를 반환합니다.

    반환:
        {
            "batch_id": int,
            "strategy": str,               # 실제 전략 (예: "Fault")
            "comparison_strategy": str,    # Golden 비교전략 (예: "APC")
            "current_time": float,
            "progress": float,              # 0~100(%)
            "deviation_score": float,       # 현재 시점 SMAPE 이탈점수(11개 변수 평균)
            "threshold": float,             # 현재 시점 임계값(평균+2.5σ)
            "status": "정상" | "주의" | "경고",
            "top3": [(변수명, 기여점수), (변수명, 기여점수), (변수명, 기여점수)],
        }
        해당 batch_id / current_time 조합이 유효하지 않으면 None.
    """
    batch_df = df[df[BATCH_COL] == batch_id].sort_values(TIME_COL)
    if batch_df.empty:
        return None

    strategy = str(batch_df[STRATEGY_COL].iloc[0])
    comparison_strategy = _get_comparison_strategy(batch_id, strategy)
    if comparison_strategy is None:
        return None

    sub = batch_df[batch_df[TIME_COL] <= current_time]
    if sub.empty:
        return None

    mean_duration = STRATEGY_MEAN_DURATION[comparison_strategy]
    current_progress = float(min(current_time / mean_duration, 1.0))

    points_mask = COMMON_POINTS <= current_progress
    points = COMMON_POINTS[points_mask]
    if len(points) == 0:
        points = COMMON_POINTS[:1]
        points_mask[0] = True

    golden_profile = golden_profiles[comparison_strategy]
    golden_ref_means = {col: golden_profile[f"{col}_mean"].to_numpy()[points_mask] for col in SMAPE_COLS}

    progress_known = np.minimum(sub[TIME_COL].to_numpy() / mean_duration, 1.0)

    per_feat_scores = {}
    for col in SMAPE_COLS:
        actual = _interpolate_to_points(progress_known, sub[col].to_numpy(), points, col)
        ref_mean = golden_ref_means[col]
        denom = np.abs(actual) + np.abs(ref_mean)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(denom < 1e-6, np.nan, np.abs(actual - ref_mean) / denom)
        per_feat_scores[col] = score

    deviation_curve = np.nanmean(np.array(list(per_feat_scores.values())), axis=0)

    thresholds = _get_thresholds(df, golden_profiles)
    threshold_curve = thresholds[comparison_strategy][points_mask]

    raw_flag = deviation_curve > threshold_curve
    warning_flag = _filter_min_run(raw_flag, MIN_RUN)
    caution_flag = raw_flag & ~warning_flag

    if warning_flag[-1]:
        status = "경고"
    elif caution_flag[-1]:
        status = "주의"
    else:
        status = "정상"

    last_idx = -1
    top3_scores = {col: float(per_feat_scores[col][last_idx]) for col in SMAPE_COLS}
    # NaN(실측값·골든평균이 둘 다 0에 가까워 SMAPE 정의 불가한 지점, 주로 레시피
    # 고정형 변수가 0인 구간)이 정렬 시 부정확하게 상위로 섞여 들어가는 것을 방지.
    # -inf로 취급해 항상 최하위로 보내는 것으로 수정 (버그 수정, 공식/판정 로직 변경 아님).
    top3 = sorted(
        top3_scores.items(),
        key=lambda x: x[1] if not np.isnan(x[1]) else float("-inf"),
        reverse=True,
    )[:3]

    return {
        "batch_id": int(batch_id),
        "strategy": strategy,
        "comparison_strategy": comparison_strategy,
        "current_time": float(current_time),
        "progress": round(current_progress * 100, 1),
        "deviation_score": float(deviation_curve[last_idx]),
        "threshold": float(threshold_curve[last_idx]),
        "status": status,
        "top3": top3,
    }


# --------------------------------------------
# PAGE2(이탈 상세진단) 화면용 확장 함수
# get_deviation_result()의 확정 로직(11개 변수 SMAPE + 전략별 2.5σ + 1지점 주의/2지점 이상 경고)은
# 그대로 재사용하고, 여기에 화면에 필요한 시계열/지속시간 정보만 추가로 계산한다.
# (SMAPE 공식·임계값·판정기준 자체는 변경하지 않음)
# --------------------------------------------

def get_deviation_history(df, golden_profiles, batch_id, current_time):
    """
    PAGE2 상세진단 화면(그래프 + 최초신호 + 지속이탈)에 필요한 시계열 정보를 반환합니다.

    반환:
        {
            "batch_id", "strategy", "comparison_strategy", "current_time", "progress",
            "points_time": 각 보간 지점의 실제 발효시간(h) 배열,
            "deviation_curve": 지점별 종합 이탈점수(SMAPE 평균) 배열,
            "threshold_curve": 지점별 임계값 배열,
            "warning_flag": 지점별 경고(연속 2지점 이상 초과) bool 배열,
            "caution_flag": 지점별 주의(1지점만 초과) bool 배열,
            "first_signal_time": 현재 이어지고 있는 이탈이 시작된 시점(h). 이탈 중이 아니면 None,
            "persist_hours": 그 시작 시점부터 현재까지 지속된 시간(h). 이탈 중이 아니면 0.0,
            "variables": {
                변수명: {"time": 실측 시간 배열, "actual": 실측값 배열,
                          "golden_mean": 골든 평균(같은 시간 배열 기준), "golden_std": 골든 표준편차}
                for 변수명 in SMAPE_COLS
            },
        }
        해당 batch_id / current_time 조합이 유효하지 않으면 None.
    """
    batch_df = df[df[BATCH_COL] == batch_id].sort_values(TIME_COL)
    if batch_df.empty:
        return None

    strategy = str(batch_df[STRATEGY_COL].iloc[0])
    comparison_strategy = _get_comparison_strategy(batch_id, strategy)
    if comparison_strategy is None:
        return None

    sub = batch_df[batch_df[TIME_COL] <= current_time]
    if sub.empty:
        return None

    mean_duration = STRATEGY_MEAN_DURATION[comparison_strategy]
    current_progress = float(min(current_time / mean_duration, 1.0))

    points_mask = COMMON_POINTS <= current_progress
    points = COMMON_POINTS[points_mask]
    if len(points) == 0:
        points = COMMON_POINTS[:1]
        points_mask[0] = True
    points_time = points * mean_duration

    golden_profile = golden_profiles[comparison_strategy]
    golden_ref_means = {col: golden_profile[f"{col}_mean"].to_numpy()[points_mask] for col in SMAPE_COLS}
    golden_ref_stds = {col: golden_profile[f"{col}_std"].to_numpy()[points_mask] for col in SMAPE_COLS}

    progress_known = np.minimum(sub[TIME_COL].to_numpy() / mean_duration, 1.0)

    per_feat_scores = {}
    per_feat_actual = {}
    for col in SMAPE_COLS:
        actual = _interpolate_to_points(progress_known, sub[col].to_numpy(), points, col)
        per_feat_actual[col] = actual
        ref_mean = golden_ref_means[col]
        denom = np.abs(actual) + np.abs(ref_mean)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(denom < 1e-6, np.nan, np.abs(actual - ref_mean) / denom)
        per_feat_scores[col] = score

    deviation_curve = np.nanmean(np.array(list(per_feat_scores.values())), axis=0)

    thresholds = _get_thresholds(df, golden_profiles)
    threshold_curve = thresholds[comparison_strategy][points_mask]

    raw_flag = deviation_curve > threshold_curve
    warning_flag = _filter_min_run(raw_flag, MIN_RUN)
    caution_flag = raw_flag & ~warning_flag

    # 현재(마지막 지점)부터 거슬러 올라가며 raw_flag가 계속 True인 구간의 시작점을 찾음
    # (get_alarm_flag/get_warning_flag과 동일하게 "연속 초과"를 기준으로 함)
    first_signal_time = None
    persist_hours = 0.0
    if raw_flag[-1]:
        start_idx = len(raw_flag) - 1
        while start_idx > 0 and raw_flag[start_idx - 1]:
            start_idx -= 1
        first_signal_time = float(points_time[start_idx])
        persist_hours = float(points_time[-1] - points_time[start_idx])

    variables = {}
    for col in SMAPE_COLS:
        variables[col] = {
            "time": points_time.tolist(),
            "actual": per_feat_actual[col].tolist(),
            "golden_mean": golden_ref_means[col].tolist(),
            "golden_std": golden_ref_stds[col].tolist(),
        }

    return {
        "batch_id": int(batch_id),
        "strategy": strategy,
        "comparison_strategy": comparison_strategy,
        "current_time": float(current_time),
        "progress": round(current_progress * 100, 1),
        "points_time": points_time.tolist(),
        "deviation_curve": deviation_curve.tolist(),
        "threshold_curve": threshold_curve.tolist(),
        "warning_flag": warning_flag.tolist(),
        "caution_flag": caution_flag.tolist(),
        "first_signal_time": first_signal_time,
        "persist_hours": persist_hours,
        "variables": variables,
    }


if __name__ == "__main__":
    # 스모크 테스트 — 정상 배치 1개 + Fault91
    from data_loader import load_data
    from golden_profile import build_all_golden_profiles

    df = load_data()
    golden_profiles = build_all_golden_profiles(df)

    print("=== 정상 배치 테스트 (Batch 1) ===")
    print(get_deviation_result(df, golden_profiles, batch_id=1, current_time=100.0))

    print("=== Fault91 / 88.6h ===")
    result = get_deviation_result(df, golden_profiles, batch_id=91, current_time=88.6)
    print(result)
    assert result["strategy"] == "Fault"
    assert result["comparison_strategy"] == "APC"
    print("스모크 테스트 통과")

    print("=== get_deviation_history / Fault91 / 88.6h ===")
    history = get_deviation_history(df, golden_profiles, batch_id=91, current_time=88.6)
    print("최초 신호 시점(h):", history["first_signal_time"])
    print("지속 이탈(h):", history["persist_hours"])
    print("포인트 수:", len(history["points_time"]))
    print("변수 목록:", list(history["variables"].keys()))
    print("PAGE2 확장 함수 스모크 테스트 통과")
