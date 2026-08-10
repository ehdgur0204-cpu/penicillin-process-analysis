# ============================================
# replay_utils.py
# ============================================

from config import (
    STRATEGY_MEAN_DURATION,
    OVERALL_MEAN_DURATION,
    FAULT_COMPARISON_STRATEGY
)

from data_loader import get_current_row

from golden_profile import get_golden_reference_at_progress


def calculate_progress(current_time, strategy):
    """
    Replay 현재 시점의 근사 공정진행률(%) 계산
    """

    if strategy in STRATEGY_MEAN_DURATION:
        mean_duration = STRATEGY_MEAN_DURATION[strategy]
    else:
        mean_duration = OVERALL_MEAN_DURATION

    progress = current_time / mean_duration * 100
    progress = max(0.0, min(progress, 100.0))

    return progress


def get_comparison_strategy(batch_id, strategy):
    """
    이탈진단 시 사용할 Golden Profile 비교 전략 반환
    """

    if strategy in ["RC", "OC", "APC"]:
        return strategy

    return FAULT_COMPARISON_STRATEGY.get(batch_id)


def get_replay_state(df, batch_id, current_time):
    """
    현재 Batch 기본 상태 정보 반환
    """

    current_row = get_current_row(
        df,
        batch_id=batch_id,
        current_time=current_time
    )

    if current_row is None:
        return None

    strategy = current_row["제어전략그룹"]
    actual_time = current_row["발효시간"]

    progress = calculate_progress(
        current_time=actual_time,
        strategy=strategy
    )

    comparison_strategy = get_comparison_strategy(
        batch_id=int(current_row["배치번호"]),
        strategy=strategy
    )

    return {
        "batch_id": int(current_row["배치번호"]),
        "strategy": str(strategy),
        "comparison_strategy": comparison_strategy,
        "current_time": float(actual_time),
        "progress": float(round(progress, 1)),
        "current_p": float(current_row["페니실린농도_P"]),
    }

def get_replay_with_golden(
    df,
    golden_profiles,
    batch_id,
    current_time
):
    """
    현재 Replay 상태와 해당 시점의 Golden 기준값을 함께 반환합니다.
    """

    state = get_replay_state(
        df,
        batch_id,
        current_time
    )

    if state is None:
        return None

    comparison_strategy = state["comparison_strategy"]

    if comparison_strategy is None:
        golden_reference = None

    else:
        # Replay progress는 % 단위이므로
        # Golden 계산용 0~1 범위로 변환
        golden_progress = state["progress"] / 100.0

        golden_reference = get_golden_reference_at_progress(
            golden_profiles,
            comparison_strategy,
            golden_progress
        )

    return {
        "state": state,
        "golden_reference": golden_reference
    }