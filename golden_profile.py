# ============================================
# golden_profile.py
# Golden Profile 생성 관련 공통 함수
# ============================================

import pandas as pd
import numpy as np

from scipy.interpolate import PchipInterpolator, interp1d

from config import (
    GOLDEN_BATCHES,
    GOLDEN_CONTINUOUS_COLS,
    GOLDEN_STEP_COLS
)

def get_golden_data(df, strategy):
    """
    특정 전략(RC/OC/APC)의 Golden Batch 데이터만 반환
    """

    if strategy not in GOLDEN_BATCHES:
        raise ValueError(
            f"지원하지 않는 전략입니다: {strategy}"
        )

    batch_ids = GOLDEN_BATCHES[strategy]

    golden_df = df[
        df["배치번호"].isin(batch_ids)
    ].copy()

    golden_df = golden_df.sort_values(
        ["배치번호", "발효시간"]
    ).reset_index(drop=True)

    return golden_df

def add_golden_progress(golden_df):
    """
    완료된 Golden Batch의 실제 최종시간을 기준으로
    공정진행률을 0~1 범위로 계산합니다.
    """

    result = golden_df.copy()

    final_time = result.groupby("배치번호")["발효시간"].transform("max")

    result["공정진행률"] = (
        result["발효시간"] / final_time
    )

    return result

def get_common_points():
    """
    Golden Profile 공통 진행률 격자 생성
    0 ~ 1, 총 101개 지점
    """

    common_points = np.linspace(0, 1, 101)

    return common_points

def interpolate_continuous_batch(batch_df, column):
    """
    Golden Batch 1개의 연속형 변수를
    0~1 공통 진행률 101개 지점에 PCHIP 보간합니다.
    """

    batch_df = add_golden_progress(batch_df)

    common_points = get_common_points()

    x = batch_df["공정진행률"].to_numpy()
    y = batch_df[column].to_numpy()

    interpolator = PchipInterpolator(x, y)

    interpolated_values = interpolator(common_points)

    return common_points, interpolated_values

def interpolate_step_batch(batch_df, column):
    """
    Golden Batch 1개의 계단형 변수를
    0~1 공통 진행률 101개 지점에
    이전값 유지(previous) 방식으로 보간합니다.
    """

    batch_df = add_golden_progress(batch_df)

    common_points = get_common_points()

    x = batch_df["공정진행률"].to_numpy()
    y = batch_df[column].to_numpy()

    interpolator = interp1d(
        x,
        y,
        kind="previous",
        bounds_error=False,
        fill_value=(y[0], y[-1])
    )

    interpolated_values = interpolator(common_points)

    return common_points, interpolated_values

def interpolate_golden_batch(batch_df):
    """
    Golden Batch 1개의 12개 공정변수를
    0~1 공통 진행률 101개 지점으로 보간합니다.

    연속형 8개 -> PCHIP
    계단형 4개 -> 이전값 유지
    """

    common_points = get_common_points()

    result = pd.DataFrame({
        "공정진행률": common_points
    })

    for column in GOLDEN_CONTINUOUS_COLS:
        _, values = interpolate_continuous_batch(
            batch_df,
            column
        )
        result[column] = values

    for column in GOLDEN_STEP_COLS:
        _, values = interpolate_step_batch(
            batch_df,
            column
        )
        result[column] = values

    return result

def build_strategy_golden_profile(df, strategy):
    """
    특정 전략의 Golden Batch들을 모두 101개 지점으로 보간한 뒤
    지점별 평균(mean)과 표준편차(std)를 계산합니다.
    """

    if strategy not in GOLDEN_BATCHES:
        raise ValueError(f"지원하지 않는 전략입니다: {strategy}")

    batch_ids = GOLDEN_BATCHES[strategy]
    batch_profiles = []

    for batch_id in batch_ids:
        batch_df = df[
            df["배치번호"] == batch_id
        ].copy()

        profile = interpolate_golden_batch(batch_df)
        batch_profiles.append(profile)

    common_points = get_common_points()

    result = pd.DataFrame({
        "공정진행률": common_points
    })

    all_columns = GOLDEN_CONTINUOUS_COLS + GOLDEN_STEP_COLS

    for column in all_columns:
        values = np.array([
            profile[column].to_numpy()
            for profile in batch_profiles
        ])

        result[f"{column}_mean"] = values.mean(axis=0)
        result[f"{column}_std"] = values.std(axis=0)

    return result

def build_all_golden_profiles(df):
    """
    RC, OC, APC 전략별 Golden Profile을 생성하여
    딕셔너리 형태로 반환합니다.
    """

    profiles = {}

    for strategy in ["RC", "OC", "APC"]:
        profiles[strategy] = build_strategy_golden_profile(
            df,
            strategy
        )

    return profiles

def get_golden_reference_at_progress(
    golden_profiles,
    strategy,
    progress
):
    """
    특정 전략의 Golden Profile에서
    주어진 공정진행률(0~1)에 해당하는
    12개 변수의 mean/std를 반환합니다.
    """

    if strategy not in golden_profiles:
        raise ValueError(
            f"Golden Profile이 없는 전략입니다: {strategy}"
        )

    progress = float(np.clip(progress, 0.0, 1.0))

    profile = golden_profiles[strategy]

    result = {
        "strategy": strategy,
        "progress": progress
    }

    all_columns = (
        GOLDEN_CONTINUOUS_COLS
        + GOLDEN_STEP_COLS
    )

    for column in all_columns:

        mean_value = np.interp(
            progress,
            profile["공정진행률"],
            profile[f"{column}_mean"]
        )

        std_value = np.interp(
            progress,
            profile["공정진행률"],
            profile[f"{column}_std"]
        )

        result[column] = {
            "mean": float(mean_value),
            "std": float(std_value)
        }

    return result