# Penicillin Process Analysis

페니실린 생산 공정 데이터를 전처리하고 탐색·시각화·모델링하기 위한 프로젝트입니다.

## 프로젝트 구조

```text
penicillin-process-analysis/
├─ data/
│  ├─ raw/          # 원본 데이터
│  ├─ interim/      # 중간 처리 데이터
│  └─ processed/    # 분석용 최종 데이터
├─ notebooks/       # 단계별 분석 노트북
├─ src/             # 재사용 가능한 Python 코드
├─ outputs/
│  ├─ figures/      # 그래프와 이미지
│  ├─ tables/       # 결과 테이블
│  └─ models/       # 학습된 모델
├─ streamlit/       # Streamlit 대시보드
└─ docs/            # 프로젝트 문서
```

## 시작하기

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Streamlit 대시보드 실행:

```powershell
streamlit run streamlit/app.py
```

## 분석 순서

1. `01_data_check.ipynb`
2. `02_preprocessing.ipynb`
3. `03_eda.ipynb`
4. `04_visualization.ipynb`
5. `05_modeling.ipynb`

## 공통 모듈

대시보드 통합에 공통으로 사용하는 데이터 로드, Replay, Golden Profile 기능입니다.

### 파일

```text
config.py
data_loader.py
replay_utils.py
golden_profile.py
merged_data_clean.csv
```

### 역할

- `config.py`
  - 공통 컬럼명
  - Golden Batch 목록
  - 전략별 평균 운전시간
  - Fault Batch의 Golden 비교 기준

- `data_loader.py`
  - 데이터 로드
  - Batch 조회
  - 현재 시점까지 데이터 추출
  - 미래 데이터 차단

- `golden_profile.py`
  - RC / OC / APC Golden Profile 생성
  - 공정진행률 0~1 정규화
  - 101개 공통 지점 보간
  - 12개 공정변수의 전략별 mean / std 제공

- `replay_utils.py`
  - Replay 진행률 계산
  - Golden 비교전략 결정
  - 현재 Replay 상태와 Golden Profile 연결

### 기본 사용법

```python
from data_loader import load_data
from golden_profile import build_all_golden_profiles
from replay_utils import get_replay_with_golden

df = load_data()

golden_profiles = build_all_golden_profiles(df)

result = get_replay_with_golden(
    df,
    golden_profiles,
    batch_id=91,
    current_time=88.6
)

print(result["state"])
print(result["golden_reference"])
```

### Batch 91 주의사항

Batch 91의 실제 데이터 구분은 `Fault`입니다.

```text
실제 구분: Fault
Golden 비교 기준: APC
```

Batch 91을 APC 공정으로 분류하는 것이 아니라,
이탈 평가를 위한 Golden 비교 기준으로 APC Profile을 사용합니다.

현재 공통 모듈은 다음 범위까지만 담당합니다.

```text
데이터 로드
→ 현재 시점 데이터 추출
→ Replay 진행률
→ Golden Profile 생성
→ 현재 시점 Golden 기준값 제공
```

SMAPE 이탈진단, 임계값, 상태 판정, Top3 진단,
12시간 후 농도예측은 각 담당자의 최종 모듈을 연결합니다.