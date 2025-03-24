import pandas as pd
from datetime import datetime, timezone, timedelta
from dateutil import parser

# KST timezone
kst = timezone(timedelta(hours=9))

def convert_to_kst(time_str):
    if not isinstance(time_str, str) or not time_str.strip():
        return pd.NaT  # 빈 문자열 또는 None 처리

    try:
        utc_dt = parser.parse(time_str)
        if utc_dt.tzinfo is None:
            # UTC 정보가 없다면 UTC로 가정
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        else:
            # 다른 timezone이 들어있으면 UTC로 변환
            utc_dt = utc_dt.astimezone(timezone.utc)
        return utc_dt.astimezone(kst)
    except Exception:
        print(f"[변환 실패] {time_str}")
        return pd.NaT

# 변환
df['kst_date'] = df['date'].apply(convert_to_kst)

# 필터링: 2021-01-01 ~ 2025-03-31 (KST 기준)
start_date = datetime(2021, 1, 1, tzinfo=kst)
end_date = datetime(2025, 3, 31, 23, 59, 59, tzinfo=kst)
filtered_df = df[(df['kst_date'] >= start_date) & (df['kst_date'] <= end_date)]

# 결과 확인
print(filtered_df)