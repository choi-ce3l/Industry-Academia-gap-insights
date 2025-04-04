import pandas as pd

# null 값 제거 함수
def drop_null_rows(df, columns_to_check):
    """
    지정한 컬럼들에 대해 null값이 있는 행을 제거하고,
    제거 전후의 데이터 개수와 컬럼별 null 개수를 출력한다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        columns_to_check (list): null 여부를 확인할 컬럼명 리스트

    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    print("📊 원본 데이터 개수:", len(df))

    for col in columns_to_check:
        null_count = df[col].isnull().sum()
        print(f"  - '{col}' 컬럼 결측치: {null_count}")

    # 결측치 제거
    df_cleaned = df.dropna(subset=columns_to_check)

    print("✅ 전처리 후 데이터 개수:", len(df_cleaned))
    return df_cleaned
