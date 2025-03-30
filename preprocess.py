import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk

# Download the 'punkt_tab' resource along with other resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')  # Download the missing resource


# ----------------------------------
# 🔧 개별 텍스트 전처리 함수
# ----------------------------------
def preprocess_text(text):
    if pd.isnull(text):
        return ""

    # HTML/XML 제거
    text = BeautifulSoup(text, "html.parser").get_text()

    # LaTeX 제거
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\\\[.*?\\\]", "", text)
    text = re.sub(r"\\\(.*?\\\)", "", text)

    # 특수기호, 숫자 제거
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 소문자 변환
    text = text.lower()

    # 토큰화
    tokens = word_tokenize(text)

    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    # 표제어 추출
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)


# ----------------------------------
# 📦 전체 파이프라인 함수
# ----------------------------------
def preprocess_and_vectorize(df, text_columns, method='count', max_features=5000):
    """
    여러 텍스트 컬럼을 전처리하고, 하나로 합쳐 벡터화까지 수행

    Parameters:
        df (pd.DataFrame): 입력 데이터
        text_columns (list): 전처리할 텍스트 컬럼 목록
        method (str): 'count' or 'tfidf'
        max_features (int): 벡터화 시 최대 단어 수

    Returns:
        vectorizer, vectorized_matrix: 훈련된 벡터라이저와 벡터 행렬
    """

    # 각 텍스트 컬럼 전처리 → 합치기
    print("🔄 전처리 중...")
    for col in text_columns:
        df[f'{col}_clean'] = df[col].apply(preprocess_text)

    df['combined_text'] = df[[f'{col}_clean' for col in text_columns]].agg(' '.join, axis=1)

    # 벡터라이저 선택
    if method == 'count':
        vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    else:
        raise ValueError("method는 'count' 또는 'tfidf' 중 하나여야 합니다.")

    print(f"✅ '{method}' 방식으로 벡터화 중...")
    vectorized_matrix = vectorizer.fit_transform(df['combined_text'])

    print("🎉 벡터화 완료! 행렬 크기:", vectorized_matrix.shape)
    return vectorizer, vectorized_matrix