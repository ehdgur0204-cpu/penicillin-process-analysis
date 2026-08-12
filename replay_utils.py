# ============================================
# replay_utils.py
#
# [수정 이력]
# calculate_progress()에 실제 strategy("Fault" 등)를 그대로 넘기면
# STRATEGY_MEAN_DURATION에 없는 키라서 OVERALL_MEAN_DURATION(228.127h)으로
# 빠지는 문제가 있었음. 하지만 이탈진단팀/ML팀 모두 Fault 배치는
# comparison_strategy(FAULT_COMPARISON_STRATEGY로 매핑된 실제 전략)의
# 평균 운전시간을 그대로 사용하는 것으로 확정(8/10 Slack 확인).
# 그래서 get_replay_state()에서 comparison_strategy를 먼저 구한 뒤
# calculate_progress()에는 comparison_strategy를 넘기도록 순서를 바꿈.
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
    Replay 현재 시점의 근사 공정진행률(%) 계산.

    주의: 여기 들어오는 strategy는 반드시 RC/OC/APC (comparison_strategy) 여야 함.
    Fault처럼 STRATEGY_MEAN_DURATION에 없는 값이 들어오면 OVERALL_MEAN_DURATION으로
    빠지는데, 이는 미매핑된 예외 상황을 위한 안전장치일 뿐 정상 경로가 아님.
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

    comparison_strategy = get_comparison_strategy(
        batch_id=int(current_row["배치번호"]),
        strategy=strategy
    )

    # 진행률은 실제 strategy가 아니라 comparison_strategy 기준으로 계산.
    # (RC/OC/APC 배치는 strategy == comparison_strategy라 결과 동일,
    #  Fault 배치만 이제 comparison_strategy의 평균 운전시간을 사용하게 됨)
    progress = calculate_progress(
        current_time=actual_time,
        strategy=comparison_strategy
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
