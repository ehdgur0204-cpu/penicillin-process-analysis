import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# ============================================================
# 기본 설정
# ============================================================

TARGET_COL = "Penicillin concentration(P:g/L)"
TIME_COL = "Time (h)"
BATCH_COL = "Batch_ID"

STEP = 0.2

FAULT_IDS = list(range(91, 101))

WINDOW_POINTS_20H = int(round(20 / STEP))


# ============================================================
# 1. 최종 24개 Feature
# ============================================================

BASE17 = [
    "Aeration rate(Fg:L/h)",
    "Sugar feed rate(Fs:L/h)",
    "Acid flow rate(Fa:L/h)",
    "Base flow rate(Fb:L/h)",
    "Heating/cooling water flow rate(Fc:L/h)",
    "Air head pressure(pressure:bar)",
    "Substrate concentration(S:g/L)",
    "Dissolved oxygen concentration(DO2:mg/L)",
    "pH(pH:pH)",
    "Temperature(T:K)",
    "Generated heat(Q:kJ)",
    "carbon dioxide percent in off-gas(CO2outgas:%)",
    "Oxygen in percent in off-gas(O2:O2  (%))",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))",
    "Carbon evolution rate(CER:g/h)",
    "Vessel Volume(V:L)",
    "Time (h)",
]

EXTRA4 = [
    "Heating water flow rate(Fh:L/h)",
    "Water for injection/dilution(Fw:L/h)",
    "Dumped broth flow(Fremoved:L/h)",
    "PAA flow(Fpaa:PAA flow (L/h))",
]

BASE21 = BASE17 + EXTRA4

FINAL_FEATURES = BASE21 + [
    "GB_score_cum_v2",
    "OUR_slope20h",
    "S_slope20h",
]

# LightGBM에 넣을 때 사용하는 안전한 컬럼명
SAFE_COLMAP = {
    c: f"f{i}"
    for i, c in enumerate(FINAL_FEATURES)
}


# ============================================================
# 2. 전략 판정
# ============================================================

FAULT_STRATEGY_REAL = {
    91: "APC",
    92: "OC",
    93: "OC",
    94: "APC",
    95: "RC",
    96: "APC",
    97: "APC",
    98: "APC",
    99: "APC",
    100: "APC",
}

GOLDEN_BATCHES = {
    "RC": [8, 12, 14, 16, 17, 26],
    "OC": [35, 48, 57],
    "APC": [62, 65, 68, 79, 82, 85],
}

CPP_11 = [
    "Penicillin concentration(P:g/L)",
    "Dissolved oxygen concentration(DO2:mg/L)",
    "pH(pH:pH)",
    "Temperature(T:K)",
    "Aeration rate(Fg:L/h)",
    "Sugar feed rate(Fs:L/h)",
    "Acid flow rate(Fa:L/h)",
    "Base flow rate(Fb:L/h)",
    "Heating/cooling water flow rate(Fc:L/h)",
    "PAA flow(Fpaa:PAA flow (L/h))",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))",
]

STEP_VARS = [
    "Aeration rate(Fg:L/h)",
    "Sugar feed rate(Fs:L/h)",
    "Acid flow rate(Fa:L/h)",
    "PAA flow(Fpaa:PAA flow (L/h))",
]

PROG_GRID = np.linspace(0, 1, 501)


def strategy_of(batch_id):
    """
    Batch ID를 기준으로 기존 최종 로직과 동일하게
    RC / OC / APC 전략을 판정한다.
    """

    if 1 <= batch_id <= 30:
        return "RC"

    if 31 <= batch_id <= 60:
        return "OC"

    if 61 <= batch_id <= 90:
        return "APC"

    if batch_id in FAULT_STRATEGY_REAL:
        return FAULT_STRATEGY_REAL[batch_id]

    raise ValueError(
        f"Batch {batch_id}: 전략을 판정할 수 없습니다 "
        "(1~100 재생 전용, 신규 배치ID 미지원)"
    )


# ============================================================
# 3. 20시간 Rolling Slope
# ============================================================

def rolling_slope_fast(
    t,
    x,
    window_points,
    min_periods=2,
):
    """
    기존 최종모델과 동일한 closed-form OLS rolling slope.
    """

    n = t.rolling(
        window_points,
        min_periods=min_periods,
    ).count()

    sum_t = t.rolling(
        window_points,
        min_periods=min_periods,
    ).sum()

    sum_x = x.rolling(
        window_points,
        min_periods=min_periods,
    ).sum()

    sum_tx = (t * x).rolling(
        window_points,
        min_periods=min_periods,
    ).sum()

    sum_tt = (t * t).rolling(
        window_points,
        min_periods=min_periods,
    ).sum()

    denom = (
        n * sum_tt
        - sum_t ** 2
    )

    return (
        n * sum_tx
        - sum_t * sum_x
    ) / denom.replace(0, np.nan)


def add_rolling_slope_full(
    df,
    col,
    out_col,
):
    """
    학습용:
    각 Batch 전체 데이터에 대해
    20시간 rolling slope를 계산한다.
    """

    df = df.copy()

    def _apply(g):
        g = g.sort_values(TIME_COL)

        return rolling_slope_fast(
            g[TIME_COL],
            g[col],
            WINDOW_POINTS_20H,
        )

    result = (
        df.groupby(
            BATCH_COL,
            group_keys=False,
        )[[TIME_COL, col]]
        .apply(_apply)
    )

    df[out_col] = result

    return df


def slope_at_current(
    sub_sorted,
    col,
):
    """
    추론용:

    이미 현재 시점까지 잘린 데이터에서
    마지막 시점의 20시간 slope 1개를 계산한다.

    따라서 현재 시점 이후의 데이터는
    함수에 들어오지 않는다.
    """

    s = rolling_slope_fast(
        sub_sorted[TIME_COL],
        sub_sorted[col],
        WINDOW_POINTS_20H,
    )

    value = s.iloc[-1]

    if pd.isna(value):
        return 0.0

    return float(value)


# ============================================================
# 4. Golden Band
# ============================================================

def smape(actual, forecast):
    """
    기존 GB_score_cum_v2에서 사용한 sMAPE 계산.
    """

    denom = (
        np.abs(actual)
        + np.abs(forecast)
    )

    denom = np.where(
        denom == 0,
        1e-9,
        denom,
    )

    return np.abs(
        actual - forecast
    ) / denom


def _interp_by_type(
    t_known,
    y_known,
    query,
    feat,
):
    """
    연속형 변수:
        PCHIP 보간

    Step 변수:
        previous-value hold
    """

    if feat in STEP_VARS:

        idx = np.searchsorted(
            t_known,
            query,
            side="right",
        ) - 1

        idx = np.clip(
            idx,
            0,
            len(t_known) - 1,
        )

        return y_known[idx]

    x_u, keep = np.unique(
        t_known,
        return_index=True,
    )

    y_u = y_known[keep]

    if len(x_u) < 2:

        return np.full_like(
            query,
            y_u[0] if len(y_u) else np.nan,
            dtype=float,
        )

    f = PchipInterpolator(
        x_u,
        y_u,
        extrapolate=False,
    )

    result = f(query)

    result = np.where(
        query < x_u[0],
        y_u[0],
        result,
    )

    result = np.where(
        query > x_u[-1],
        y_u[-1],
        result,
    )

    return result


def _build_band(
    df_train,
    batch_list,
    dur,
):
    """
    지정된 Golden Batch들의 진행률 기준 평균 곡선을 생성한다.
    """

    curves = {
        feat: []
        for feat in CPP_11
    }

    for batch_id in batch_list:

        sub = (
            df_train[
                df_train[BATCH_COL] == batch_id
            ]
            .sort_values(TIME_COL)
        )

        progress = (
            sub[TIME_COL] / dur
        ).clip(
            upper=1.0
        ).to_numpy()

        for feat in CPP_11:

            curves[feat].append(
                _interp_by_type(
                    progress,
                    sub[feat].to_numpy(),
                    PROG_GRID,
                    feat,
                )
            )

    return {
        feat: np.nanmean(
            np.array(values),
            axis=0,
        )
        for feat, values in curves.items()
    }


def fit_golden_bands(
    df_train,
    duration_override=None,
):
    """
    최종모델 학습에서 사용한 Golden Band 생성.

    Streamlit 예측에서는 새로 학습하지 않고,
    final_model_bundle.joblib에 저장된 artifact를 사용한다.
    """

    if duration_override is not None:

        strategy_mean_duration = dict(
            duration_override
        )

    else:

        normal90 = df_train[
            df_train[BATCH_COL] <= 90
        ].copy()

        normal90["_strategy"] = (
            normal90[BATCH_COL]
            .map(strategy_of)
        )

        strategy_mean_duration = (
            normal90
            .groupby("_strategy")
            .apply(
                lambda g:
                g.groupby(BATCH_COL)[TIME_COL]
                .max()
                .mean()
            )
            .to_dict()
        )

    full_bands = {
        strategy: _build_band(
            df_train,
            batches,
            strategy_mean_duration[strategy],
        )
        for strategy, batches in GOLDEN_BATCHES.items()
    }

    loo_bands = {
        strategy: {
            batch_id: _build_band(
                df_train,
                [
                    x
                    for x in batches
                    if x != batch_id
                ],
                strategy_mean_duration[strategy],
            )
            for batch_id in batches
        }
        for strategy, batches in GOLDEN_BATCHES.items()
    }

    return {
        "strategy_mean_duration":
            strategy_mean_duration,

        "full_bands":
            full_bands,

        "loo_bands":
            loo_bands,

        "prog_grid":
            PROG_GRID,
    }


def _get_band(
    artifact,
    batch_id,
    strategy,
):
    """
    Golden Batch 자체를 평가할 경우 LOO band,
    그 외에는 full band 사용.
    """

    if (
        strategy in GOLDEN_BATCHES
        and batch_id in GOLDEN_BATCHES[strategy]
    ):
        return artifact[
            "loo_bands"
        ][strategy][batch_id]

    return artifact[
        "full_bands"
    ][strategy]


def gb_score_cum_v2_curve(
    artifact,
    sub_sorted,
    batch_id,
):
    """
    현재까지의 데이터에 대해
    GB_score_cum_v2를 시점별로 계산한다.

    마지막 값이 현재 시점까지의
    누적 Golden Band 이탈점수이다.
    """

    strategy = strategy_of(batch_id)

    band = _get_band(
        artifact,
        batch_id,
        strategy,
    )

    duration = artifact[
        "strategy_mean_duration"
    ][strategy]

    progress = (
        sub_sorted[TIME_COL] / duration
    ).clip(
        upper=1.0
    ).to_numpy()

    per_feature = [
        smape(
            sub_sorted[feat].to_numpy(),
            np.interp(
                progress,
                artifact["prog_grid"],
                band[feat],
            ),
        )
        for feat in CPP_11
    ]

    gb_score = np.mean(
        np.stack(per_feature, axis=0),
        axis=0,
    )

    return np.cumsum(gb_score)


def gb_score_cum_v2_full(
    df,
    artifact,
):
    """
    학습용:
    전체 데이터에 대해
    GB_score_cum_v2를 생성한다.
    """

    def _one_batch(batch_id):

        sub = (
            df[
                df[BATCH_COL] == batch_id
            ]
            .sort_values(TIME_COL)
        )

        return pd.Series(
            gb_score_cum_v2_curve(
                artifact,
                sub,
                batch_id,
            ),
            index=sub.index,
        )

    parts = [
        _one_batch(batch_id)
        for batch_id
        in sorted(
            df[BATCH_COL].unique()
        )
    ]

    return pd.concat(
        parts
    ).sort_index()


def gb_score_cum_v2_at_current(
    artifact,
    sub_sorted,
    batch_id,
):
    """
    추론용:

    현재 시점까지 잘린 데이터에서
    현재 시점의 GB_score_cum_v2 1개를 계산한다.
    """

    values = gb_score_cum_v2_curve(
        artifact,
        sub_sorted,
        batch_id,
    )

    return float(values[-1])