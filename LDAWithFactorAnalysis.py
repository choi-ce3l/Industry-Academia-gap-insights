import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import FactorAnalysis
from gensim import corpora, models

# ----------------------------------
# 🔧 텍스트 전처리 함수
# ----------------------------------
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    if isinstance(text, list):
        text = ' '.join(text)
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
    text_columns = ['title', 'abstract', 'keywords'] if data_type == 'journal' else ['title', 'content', 'keywords']
    for col in text_columns:
        if col not in df.columns:
            df[col] = ''
        df[f'{col}_clean'] = df[col].apply(preprocess_text)

    df['combined_text'] = df[[f'{col}_clean' for col in text_columns]].agg(' '.join, axis=1)
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english', max_df=0.90) if method == 'count' else TfidfVectorizer(max_features=max_features, stop_words='english', max_df=0.90)
    vectorized_matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, vectorized_matrix

# ----------------------------------
# 🔍 문서 준비
# ----------------------------------
def prepare_documents(df, data_type='journal'):
    df = df.fillna('')
    text_columns = ['title', 'abstract', 'keywords'] if data_type == 'journal' else ['title', 'content', 'keywords']
    for col in text_columns:
        if col not in df.columns:
            df[col] = ''
    df['combined_text'] = df[text_columns].agg(' '.join, axis=1)
    df['combined_text'] = df['combined_text'].apply(preprocess_text)
    return [doc.split() for doc in df['combined_text']]

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
# 📊 최적 요인 수 자동 선택
# ----------------------------------
def find_optimal_factors(topic_df, max_factors=10):
    scores = []
    for n in range(2, max_factors + 1):
        fa = FactorAnalysis(n_components=n, random_state=42)
        fa.fit(topic_df)
        scores.append(fa.score(topic_df))
    return scores

def run_factor_analysis_with_auto_selection(topic_df, max_factors=10, plot=False):
    scores = find_optimal_factors(topic_df, max_factors)
    optimal_factors = np.argmax(scores) + 2
    if plot:
        plt.plot(range(2, max_factors + 1), scores, marker='o')
        plt.xlabel("Number of Factors")
        plt.ylabel("Log-Likelihood")
        plt.title("Factor Selection via Log-Likelihood")
        plt.grid(True)
        plt.show()
    fa = FactorAnalysis(n_components=optimal_factors, random_state=42)
    factors = fa.fit_transform(topic_df)
    loadings = pd.DataFrame(fa.components_.T, index=topic_df.columns, columns=[f"Factor_{i+1}" for i in range(optimal_factors)])
    return pd.DataFrame(factors, columns=loadings.columns), loadings, optimal_factors

# ----------------------------------
# 📄 상위 문서 저장
# ----------------------------------
def top_docs_by_factor(factor_df, docs_df, top_n=5, output_path='top_docs.txt', data_type='journal'):
    with open(output_path, 'w', encoding='utf-8') as f:
        for factor in factor_df.columns:
            f.write(f"\n📌 상위 문서 - {factor}\n{'='*60}\n")
            top_indices = factor_df[factor].nlargest(top_n).index
            for i in top_indices:
                row = docs_df.loc[i]
                f.write(f"📆 Date: {row.get('date', '')}\n")
                f.write(f"📄 Title: {row.get('title', '')}\n")
                content = row.get('abstract' if data_type == 'journal' else 'content', '')
                if len(content) > 500:
                    content = content[:500].rstrip() + "..."
                f.write(f"🔍 Content (500자 이내): {content}\n")
                f.write(f"🏷️ Keywords: {row.get('keywords', '')}\n")
                f.write(f"{'-'*60}\n")
    print(f"✅ 저장 완료: {output_path}")

# ----------------------------------
# 📝 토픽 키워드 저장
# ----------------------------------
def save_lda_topics(lda_model, num_words, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, topic in lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False):
            keywords = ", ".join([word for word, _ in topic])
            f.write(f"Topic {idx}: {keywords}\n")
    print(f"✅ LDA 토픽 저장 완료: {output_path}")

# ----------------------------------
# 🎯 LDA + Factor 자동 분석 전체 파이프라인
# ----------------------------------
def run_yearly_lda_factor_analysis(df, data_type='journal', n_topics=10, max_factors=10, vectorizer_method='tfidf', max_features=5000, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    years = sorted(df['date'].unique())

    for year in years:
        print(f"\n🔵 Processing year: {year}")
        year_df = df[df['date'] == year].reset_index(drop=True)

        vectorizer, vectorized_matrix = preprocess_and_vectorize(year_df, method=vectorizer_method, max_features=max_features, data_type=data_type)

        processed_docs = prepare_documents(year_df, data_type=data_type)
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=42, passes=10)

        save_lda_topics(lda_model, num_words=10, output_path=f"{output_dir}/02_{data_type}_{year}_lda_topics.txt")

        topic_df = extract_topic_matrix(lda_model, corpus, n_topics)

        factor_df, loadings, selected_n_factors = run_factor_analysis_with_auto_selection(topic_df, max_factors=max_factors, plot=True)

        topic_df.to_csv(f"{output_dir}/02_{data_type}_{year}_topic_matrix.csv", index=False)
        factor_df.to_csv(f"{output_dir}/02_{data_type}_{year}_factor_scores.csv", index=False)
        loadings.to_csv(f"{output_dir}/02_{data_type}_{year}_factor_loadings.csv", index=True)

        top_docs_by_factor(factor_df, year_df, top_n=5, output_path=f"{output_dir}/02_{data_type}_{year}_top_docs_by_factor.txt", data_type=data_type)

        print(f"✅ Year {year} 완료! (요인 수: {selected_n_factors})")


'''# 실행 예시
run_yearly_lda_factor_analysis(
    df=df,                # 🔸 DataFrame에 'date', 'title', 'abstract/content', 'keywords' 필요
    data_type='journal',            # 🔹 'journal' 또는 'article'
    n_topics=20,                    # 🔸 LDA 토픽 수
    max_factors=10,                 # 🔸 요인 분석 시 최대 요인 수
    vectorizer_method='tfidf',     # 🔸 'count' 또는 'tfidf'
    output_dir='data/02_journal'            # 🔸 결과 저장 디렉토리
)'''