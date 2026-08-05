"""
골든배치 프로젝트 공통 모듈
- 컬럼명 변수
- 골든배치 최종 15개 + 프로파일(bands)
- Train/Test 배치 분할 (고정된 리스트)

사용법: 노트북에서 아래처럼 불러오기
    from golden_batch_common import *
"""

import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

# ── 1. 컬럼명 변수 ──────────────────────────────
batch_col = '배치번호'
time_col = '발효시간'
strategy_col = '제어전략그룹'
p_col = '페니실린농도_P'
s_col = '기질농도_S'

continuous_cols = [p_col, s_col, '용존산소_DO2', 'pH', '발효온도_T', '산소소비율_OUR',
                   '염기투입유량_Fb', '냉난방수유량_Fc']
step_cols = ['공기주입유량_Fg', '당공급유량_Fs', '산투입유량_Fa', 'PAA투입유량_Fpaa']
feature_cols = continuous_cols + step_cols

# ── 2. 데이터 불러오기 ──────────────────────────
df = pd.read_csv('merged_data_clean.csv')

# ── 3. 골든배치 최종 확정 (15개, 정렬된 순서) ──────
golden_batches = {
    'RC':  [8, 12, 14, 16, 17, 26],
    'OC':  [35, 48, 57],
    'APC': [62, 65, 68, 79, 82, 85]
}

# ── 4. 골든배치 기준 프로파일(bands) 생성 함수 ─────
def build_golden_band(df, batch_list, continuous_cols, step_cols, time_col, n_points=101):
    common_points = np.linspace(0, 1, n_points)
    result = {}
    all_cols = continuous_cols + step_cols
    for col in all_cols:
        values = []
        for b in batch_list:
            sub = df[df[batch_col] == b].sort_values(time_col)
            progress = sub[time_col] / sub[time_col].max()
            if col in continuous_cols:
                interpolator = PchipInterpolator(progress, sub[col])
                interpolated = interpolator(common_points)
            else:
                interpolator = interp1d(progress, sub[col], kind='previous',
                                         bounds_error=False, fill_value=(sub[col].iloc[0], sub[col].iloc[-1]))
                interpolated = interpolator(common_points)
            values.append(interpolated)
        values = np.array(values)
        result[f'{col}_mean'] = values.mean(axis=0)
        result[f'{col}_std'] = values.std(axis=0)
    band = pd.DataFrame(result, index=common_points)
    band.index.name = '공정진행률'
    return band.reset_index()

bands = {}
for strategy, batches in golden_batches.items():
    bands[strategy] = build_golden_band(df, batches, continuous_cols, step_cols, time_col)

# ── 5. Test/Train 배치 분할 (고정 리스트 — 이미 확인된 결과를 그대로 저장) ──
# GroupShuffleSplit(random_state=42)로 이미 한 번 나눠서 확인한 결과를 그대로 고정
# (매번 다시 split 돌리지 않음 — 환경마다 결과가 미묘하게 달라질 위험 방지)
test_batches_all = [1, 5, 11, 13, 19, 23, 31, 32, 34, 40, 45, 46, 54, 71, 74, 77, 78, 81, 84, 91]
test_batches_normal = [b for b in test_batches_all if b != 91]  # 91(Fault) 제외 19개

print("golden_batch_common 모듈 로드 완료")
print(f"골든배치: RC {len(golden_batches['RC'])}개, OC {len(golden_batches['OC'])}개, APC {len(golden_batches['APC'])}개")
print(f"Test 배치: 전체 {len(test_batches_all)}개, 정상 {len(test_batches_normal)}개")