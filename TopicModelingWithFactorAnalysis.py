import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import FactorAnalysis
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import os

# ----------------------------------
# 🔧 텍스트 전처리 함수
# ----------------------------------
def preprocess_text(text):
    if pd.isnull(text):
        return ""

    if isinstance(text, list):
        text = ' '.join(text)

    # HTML/XML 제거
    if isinstance(text, str) and ('<' in text and '>' in text):
        text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\\\(.*?\\\)", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

# ----------------------------------
# 📦 전처리 및 벡터화
# ----------------------------------
def preprocess_and_vectorize(df, method='count', max_features=5000, data_type='journal'):
    if data_type == 'journal':
        text_columns = ['title', 'abstract', 'keywords']
    elif data_type == 'article':
        text_columns = ['title', 'content', 'keywords']
    else:
        raise ValueError("data_type은 'journal' 또는 'article' 중 하나여야 합니다.")

    for col in text_columns:
        if col not in df.columns:
            df[col] = ''
        df[f'{col}_clean'] = df[col].apply(preprocess_text)

    df['combined_text'] = df[[f'{col}_clean' for col in text_columns]].agg(' '.join, axis=1)

    if method == 'count':
        vectorizer = CountVectorizer(max_features=max_features, stop_words='english', max_df=0.90)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', max_df=0.90)
    else:
        raise ValueError("method는 'count' 또는 'tfidf' 중 하나여야 합니다.")

    vectorized_matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, vectorized_matrix

# ----------------------------------
# 🔍 문서 준비 및 토큰화
# ----------------------------------
def prepare_documents(df, data_type='journal'):
    df = df.fillna('')

    if 'combined_text' not in df.columns:
        if data_type == 'journal':
            text_columns = ['title', 'abstract', 'keywords']
        elif data_type == 'article':
            text_columns = ['title', 'content', 'keywords']
        else:
            raise ValueError("data_type은 'journal' 또는 'article' 중 하나여야 합니다.")

        for col in text_columns:
            if col not in df.columns:
                df[col] = ''
        df['combined_text'] = df[text_columns].agg(' '.join, axis=1)
        df['combined_text'] = df['combined_text'].apply(preprocess_text)

    return [doc.split() for doc in df['combined_text']]

# ----------------------------------
# 📈 Coherence 점수 계산
# (여기서는 안쓰지만 남겨둠)
# ----------------------------------
def compute_coherence_scores(dictionary, corpus, texts, start, limit, step):
    scores = []
    for k in range(start, limit, step):
        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=42, passes=10)
        cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
        scores.append((k, cm.get_coherence()))
    return scores

# ----------------------------------
# 🧩 토픽-문서 행렬 생성
# ----------------------------------
def extract_topic_matrix(lda_model, corpus, num_topics):
    topic_matrix = []
    for doc in corpus:
        dist = lda_model.get_document_topics(doc, minimum_probability=0)
        topic_matrix.append([prob for _, prob in sorted(dist)])
    return pd.DataFrame(topic_matrix, columns=[f"Topic_{i}" for i in range(num_topics)])

# ----------------------------------
# 📊 요인 분석
# ----------------------------------
def run_factor_analysis(topic_df, n_factors=5, max_iter=500):
    fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=max_iter)
    factors = fa.fit_transform(topic_df)
    loadings = pd.DataFrame(fa.components_.T, index=topic_df.columns, columns=[f"Factor_{i+1}" for i in range(n_factors)])
    return pd.DataFrame(factors, columns=loadings.columns), loadings

# ----------------------------------
# 📊 요인 분석 상위 문서 5개 저장
# ----------------------------------
# ----------------------------------
# 📊 요인 분석 상위 문서 5개 저장 (500글자 제한 추가)
# ----------------------------------
def top_docs_by_factor(factor_df, docs_df, top_n=5, output_path='top_documents_by_factor.txt', data_type='journal'):
    with open(output_path, 'w', encoding='utf-8') as f:
        for factor in factor_df.columns:
            f.write(f"\n📌 상위 문서 - {factor}\n")
            f.write("="*60 + "\n")
            top_indices = factor_df[factor].nlargest(top_n).index

            for i in top_indices:
                row = docs_df.loc[i]

                f.write(f"📆 Date: {row.get('date', '')}\n")
                f.write(f"📄 Title: {row.get('title', '')}\n")

                if data_type == 'journal':
                    content = row.get('abstract', '')
                elif data_type == 'article':
                    content = row.get('content', '')
                else:
                    raise ValueError("data_type은 'journal' 또는 'article' 중 하나여야 합니다.")

                # 🔵 본문 500글자까지만 저장
                if len(content) > 500:
                    content = content[:500].rstrip() + "..."

                f.write(f"🔍 Content (500자 이내): {content}\n")
                f.write(f"🏷️ Keywords: {row.get('keywords', '')}\n")
                f.write("-"*60 + "\n")

    print(f"✅ 저장 완료 (본문 500자 제한 적용): {output_path}")

# ----------------------------------
# 📝 LDA 모델로부터 토픽별 키워드 저장
# ----------------------------------
def save_lda_topics(lda_model, num_words, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, topic in lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False):
            keywords = ", ".join([word for word, _ in topic])
            f.write(f"Topic {idx}: {keywords}\n")
    print(f"✅ LDA 토픽 저장 완료: {output_path}")

# ----------------------------------
# 🎯 년도별로 LDA + Factor 분석
# ----------------------------------
def run_yearly_lda_factor_analysis(df, data_type='journal', n_topics=10, n_factors=5, vectorizer_method='tfidf', max_features=5000, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    years = sorted(df['date'].unique())

    for year in years:
        print(f"🔵 Processing year: {year}")

        # 해당 연도 데이터 추출
        year_df = df[df['date'] == year].reset_index(drop=True)

        # 벡터화
        vectorizer, vectorized_matrix = preprocess_and_vectorize(year_df, method=vectorizer_method, max_features=max_features, data_type=data_type)

        # Gensim LDA용 corpus 준비
        processed_docs = prepare_documents(year_df, data_type=data_type)
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        # LDA 모델 학습
        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=42, passes=10)

        # 🎯 LDA 토픽별 키워드 저장 추가
        save_lda_topics(
            lda_model=lda_model,
            num_words=10,  # 토픽당 상위 10개 단어
            output_path=f"{output_dir}/02_{data_type}_{year}_lda_topics.txt"
        )

        # 토픽-문서 행렬 생성
        topic_df = extract_topic_matrix(lda_model, corpus, n_topics)

        # Factor Analysis
        factor_df, loadings = run_factor_analysis(topic_df, n_factors=n_factors)

        # 결과 저장
        topic_df.to_csv(f"{output_dir}/02_{data_type}_{year}_topic_matrix.csv", index=False)
        factor_df.to_csv(f"{output_dir}/02_{data_type}{year}_factor_scores.csv", index=False)
        loadings.to_csv(f"{output_dir}/02_{data_type}{year}_factor_loadings.csv", index=True)

        top_docs_by_factor(
            factor_df=factor_df,
            docs_df=year_df,
            top_n=5,
            output_path=f"{output_dir}/02_{data_type}_{year}_top_docs_by_factor.txt",  # 여기 수정
            data_type=data_type
        )

        print(f"✅ Year {year} 완료! (Topic Matrix, Factor Scores, Loadings, Top Docs 저장)")


''' 사용 방법
run_yearly_lda_factor_analysis(
    df,  # 👉 당신이 만든 데이터프레임
    data_type='journal',  # 👉 논문이면 'journal', 뉴스 기사면 'article'
    n_topics=10,          # 👉 고정: LDA 토픽 개수
    n_factors=5,          # 👉 고정: Factor Analysis 요인 수
    vectorizer_method='tfidf',  # 👉 'count'나 'tfidf' 중 선택 ('tfidf' 추천)
    max_features=5000,     # 👉 벡터라이저 최대 단어 수 (선택사항)
    output_dir='data/result/02'   # 👉 결과 저장 폴더명
)
'''