import sys
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

# src 폴더의 feature_engineering.py 사용
sys.path.insert(0, "src")
import feature_engineering as fe


# ============================================================
# 설정
# ============================================================

DATA_PATH = "data/raw/merged_data.csv"
MODEL_PATH = "final_model_bundle.joblib"

H = 12
STEPS = round(H / fe.STEP)

# 대시보드팀과 통일한 Train 정상71 기준 운전시간
DASHBOARD_DURATION = {
    "RC": 232.125,
    "OC": 226.739,
    "APC": 225.458,
}


# ============================================================
# 메인
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. 데이터 로드
    # --------------------------------------------------------
    print("1) 데이터 로드")

    df = (
        pd.read_csv(DATA_PATH)
        .sort_values(
            [fe.BATCH_COL, fe.TIME_COL]
        )
        .reset_index(drop=True)
    )

    print(
        f"   shape: {df.shape}"
        f" / 배치 수: {df[fe.BATCH_COL].nunique()}"
    )


    # --------------------------------------------------------
    # 2. Golden Band 생성
    # --------------------------------------------------------
    print(
        "2) Golden Band artifact 생성"
    )

    artifact = fe.fit_golden_bands(
        df,
        duration_override=DASHBOARD_DURATION,
    )

    print(
        "   strategy_mean_duration:"
    )

    for strategy, duration in artifact[
        "strategy_mean_duration"
    ].items():

        print(
            f"   {strategy}: {duration:.3f} h"
        )


    # --------------------------------------------------------
    # 3. GB_score_cum_v2 생성
    # --------------------------------------------------------
    print(
        "3) GB_score_cum_v2 계산"
    )

    df["GB_score_cum_v2"] = (
        fe.gb_score_cum_v2_full(
            df,
            artifact,
        )
    )


    # --------------------------------------------------------
    # 4. 20시간 slope 생성
    # --------------------------------------------------------
    print(
        "4) OUR_slope20h / S_slope20h 계산"
    )

    df = fe.add_rolling_slope_full(
        df,
        "Substrate concentration(S:g/L)",
        "S_slope20h",
    )

    df = fe.add_rolling_slope_full(
        df,
        "Oxygen Uptake Rate(OUR:(g min^{-1}))",
        "OUR_slope20h",
    )

    df[
        [
            "S_slope20h",
            "OUR_slope20h",
        ]
    ] = df[
        [
            "S_slope20h",
            "OUR_slope20h",
        ]
    ].fillna(0)


    # --------------------------------------------------------
    # 5. 정확히 12시간 후 Target 생성
    # --------------------------------------------------------
    print(
        "5) 12시간 후 Target 생성"
    )

    df["target_12h"] = (
        df.groupby(fe.BATCH_COL)[
            fe.TARGET_COL
        ].shift(-STEPS)
    )

    data = (
        df
        .dropna(subset=["target_12h"])
        .reset_index(drop=True)
    )

    print(
        f"   STEP = {STEPS}"
        f" / STEP 시간 = {fe.STEP}h"
    )

    print(
        f"   학습 데이터 행 수: {len(data)}"
    )


    # --------------------------------------------------------
    # 6. 최종 24개 Feature 구성
    # --------------------------------------------------------
    print(
        "6) 최종 24개 Feature 구성"
    )

    X = (
        data[
            fe.FINAL_FEATURES
        ]
        .rename(
            columns=fe.SAFE_COLMAP
        )
    )

    y = data["target_12h"]


    print(
        f"   Feature 수: {X.shape[1]}"
    )


    # --------------------------------------------------------
    # 7. LightGBM 최종모델 학습
    # --------------------------------------------------------
    print(
        "7) LightGBM 최종모델 학습"
    )

    model = lgb.LGBMRegressor(
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )

    model.fit(X, y)


    # --------------------------------------------------------
    # 8. In-sample 참고 성능
    # --------------------------------------------------------
    pred = model.predict(X)

    r2 = r2_score(
        y,
        pred,
    )

    mae = mean_absolute_error(
        y,
        pred,
    )

    print(
        f"   In-sample R2  : {r2:.4f}"
    )

    print(
        f"   In-sample MAE : {mae:.4f}"
    )

    print(
        "   ※ 위 수치는 학습 데이터 자체에"
        " 대한 참고용 성능입니다."
    )


    # --------------------------------------------------------
    # 9. 모델 + Feature + Golden Band 저장
    # --------------------------------------------------------
    print(
        "8) 최종 모델 bundle 저장"
    )

    bundle = {
        "model": model,

        "golden_band_artifact": artifact,

        "feature_columns":
            fe.FINAL_FEATURES,

        "safe_colmap":
            fe.SAFE_COLMAP,

        "horizon_h":
            H,

        "step":
            fe.STEP,

        "trained_on":
            "all_100_batches",
    }

    joblib.dump(
        bundle,
        MODEL_PATH,
    )


    print()
    print("=" * 60)
    print("완료!")
    print("=" * 60)
    print(
        f"저장 위치: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()