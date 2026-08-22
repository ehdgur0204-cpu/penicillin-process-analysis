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
