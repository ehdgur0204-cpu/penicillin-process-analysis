import joblib
import pandas as pd

import feature_engineering as fe


def predict_12h(
    df,
    batch_id,
    current_time,
    model_path="final_model_bundle.joblib",
):
    """
    현재 Batch의 현재 시점까지의 데이터만 사용하여
    정확히 12시간 후 Penicillin concentration을 예측한다.

    Parameters
    ----------
    df : pandas.DataFrame
        전체 공정 데이터
    batch_id : int
        예측 대상 Batch ID
    current_time : float
        현재 공정 시간(h)
    model_path : str
        최종 모델 bundle 경로

    Returns
    -------
    dict
        batch_id
        current_time
        target_time
        current_p
        predicted_p_12h
        change
    """

    # --------------------------------------------------
    # 1. 최종 모델 bundle 불러오기
    # --------------------------------------------------
    bundle = joblib.load(model_path)

    model = bundle["model"]
    artifact = bundle["golden_band_artifact"]
    feature_columns = bundle["feature_columns"]
    safe_colmap = bundle["safe_colmap"]
    horizon_h = bundle["horizon_h"]

    # --------------------------------------------------
    # 2. 해당 Batch만 선택
    # --------------------------------------------------
    batch_df = df[
        df[fe.BATCH_COL] == batch_id
    ].copy()

    if batch_df.empty:
        raise ValueError(
            f"Batch {batch_id}의 데이터를 찾을 수 없습니다."
        )

    # 시간순 정렬
    batch_df = (
        batch_df
        .sort_values(fe.TIME_COL)
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 3. 현재 시점까지 데이터만 사용
    # --------------------------------------------------
    current_df = batch_df[
        batch_df[fe.TIME_COL] <= current_time
    ].copy()

    if current_df.empty:
        raise ValueError(
            f"Batch {batch_id}에서 "
            f"{current_time}h 이전의 데이터가 없습니다."
        )

    current_df = (
        current_df
        .sort_values(fe.TIME_COL)
        .reset_index(drop=True)
    )

    # 실제 사용한 마지막 관측 시점
    actual_current_time = float(
        current_df[fe.TIME_COL].iloc[-1]
    )

    # --------------------------------------------------
    # 4. 현재 농도
    # --------------------------------------------------
    current_p = float(
        current_df[fe.TARGET_COL].iloc[-1]
    )

    # --------------------------------------------------
    # 5. GB_score_cum_v2
    #
    # 저장된 Golden Band artifact를 사용한다.
    # 현재 시점까지 잘린 데이터만 전달한다.
    # --------------------------------------------------
    gb_score = fe.gb_score_cum_v2_at_current(
        artifact=artifact,
        sub_sorted=current_df,
        batch_id=batch_id,
    )

    # 현재 시점 행에 GB score 입력
    current_df["GB_score_cum_v2"] = gb_score

    # --------------------------------------------------
    # 6. 20시간 slope
    #
    # 현재 시점까지의 데이터만 사용
    # --------------------------------------------------
    s_slope = fe.slope_at_current(
        current_df,
        "Substrate concentration(S:g/L)",
    )

    our_slope = fe.slope_at_current(
        current_df,
        "Oxygen Uptake Rate(OUR:(g min^{-1}))",
    )

    current_df["S_slope20h"] = s_slope
    current_df["OUR_slope20h"] = our_slope

    # --------------------------------------------------
    # 7. 현재 시점의 마지막 행을 Feature로 사용
    # --------------------------------------------------
    feature_row = current_df.iloc[-1:].copy()

    # 최종 학습 때 사용한 24개 Feature
    X = feature_row[
        feature_columns
    ].copy()

    # 학습 때 사용한 f0 ~ f23 컬럼명으로 변환
    X = X.rename(
        columns=safe_colmap
    )

    # --------------------------------------------------
    # 8. LightGBM으로 12시간 후 농도 예측
    # --------------------------------------------------
    predicted_p_12h = float(
        model.predict(X)[0]
    )

    # --------------------------------------------------
    # 9. 현재 농도 대비 변화량
    # --------------------------------------------------
    change = predicted_p_12h - current_p

    # --------------------------------------------------
    # 10. 결과 반환
    # --------------------------------------------------
    return {
        "batch_id": int(batch_id),
        "current_time": actual_current_time,
        "target_time": actual_current_time + horizon_h,
        "current_p": current_p,
        "predicted_p_12h": predicted_p_12h,
        "change": change,
    }