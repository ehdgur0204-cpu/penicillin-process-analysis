import pandas as pd


DATA_PATH = "merged_data_clean.csv"


def load_data():
    """
    전처리가 완료된 전체 공정 데이터를 불러옵니다.
    """
    df = pd.read_csv(DATA_PATH)

    return df

def get_batch(df, batch_id):
    """
    특정 Batch 데이터를 발효시간 순서로 반환합니다.
    """
    batch_df = df[df["배치번호"] == batch_id].copy()
    batch_df = batch_df.sort_values("발효시간").reset_index(drop=True)

    return batch_df

def get_batch_until_time(df, batch_id, current_time):
    """
    특정 Batch에서 현재 시점(current_time)까지의 데이터만 반환합니다.
    미래 시점 데이터는 포함하지 않습니다.
    """
    batch_df = get_batch(df, batch_id)

    current_df = batch_df[
        batch_df["발효시간"] <= current_time
    ].copy()

    current_df = current_df.reset_index(drop=True)

    return current_df

def get_current_row(df, batch_id, current_time):
    """
    특정 Batch의 current_time까지 관측된 데이터 중
    가장 최근 시점의 한 행을 반환합니다.
    """
    current_df = get_batch_until_time(
        df,
        batch_id=batch_id,
        current_time=current_time
    )

    if current_df.empty:
        return None

    return current_df.iloc[-1]