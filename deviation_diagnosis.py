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
    top3 = sorted(top3_scores.items(), key=lambda x: x[1], reverse=True)[:3]

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
