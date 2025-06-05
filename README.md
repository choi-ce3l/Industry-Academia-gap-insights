# AI_news 프로젝트

이 저장소는 주요 학회와 미디어 사이트에서 인공지능 관련 자료를 수집하고 전처리하여 토픽 모델링을 수행하기 위한 스크립트를 모아둔 곳입니다. `main` 브랜치에는 크롤러와 전처리, 토픽 분석에 필요한 파이썬 스크립트와 노트북이 포함되어 있습니다.

## 포함된 주요 스크립트

- **크롤러**
  - `HICSS_crawler.py` – HICSS 학회 논문 정보를 페이지 단위로 수집합니다.
  - `ICIS_crawling.py` – ICIS 학회 논문을 다운로드합니다.
  - `misq_crawler.py` – MISQ 저널 사이트에서 논문 정보를 가져옵니다.
  - `techcrunch_crawler.py` – TechCrunch AI 관련 기사 크롤링 예제입니다.
  - `theverge_news_crawl.py` – The Verge 사이트에서 AI 기사 목록과 내용을 수집합니다.

- **전처리 및 분석**
  - `preprocess.py`, `preprocessing.ipynb` – 텍스트 정제 및 벡터화 도구.
  - `LDAWithFactorAnalysis.py`, `LSAwithFactorAnalysis.py` – LDA 및 LSA 기반 토픽 모델링 후 요인 분석을 수행합니다.
  - `remove_null_rows.py`, `drop_null_rows.py` – 결측 데이터 제거 스크립트.
  - `UTC_to_KST.py` – UTC 시간을 한국 표준시로 변환합니다.

- **노트북**
  - `01_토픽모델링_산업.ipynb`, `01_토픽모델링_저널.ipynb` – 수집된 데이터로 토픽 모델링을 수행하는 예제 노트북입니다.

## 기본 사용 방법

1. 필요한 파이썬 패키지를 설치합니다.
   ```bash
   pip install -r requirements.txt  # 파일이 존재할 경우
   ```
2. 원하는 크롤러 스크립트를 실행하여 데이터를 수집합니다.
   ```bash
   python theverge_news_crawl.py --start 2023-01-01 --end 2023-01-31 --output data.csv
   ```
3. 수집된 데이터를 `preprocess.py` 등으로 전처리한 뒤 LDA/LSA 스크립트로 토픽 분석을 진행합니다.

## 디렉토리 구조 예시

```
AI_news/
├── HICSS_crawler.py
├── ICIS_crawling.py
├── LDAWithFactorAnalysis.py
├── ...
└── theverge_news_crawl.py
```

각 스크립트의 세부 사용 방법과 옵션은 파일 내부의 주석을 참고하세요.

## 라이선스

별도의 명시가 없는 한 본 프로젝트의 코드는 MIT 라이선스를 따릅니다.

