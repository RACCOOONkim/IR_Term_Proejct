import os
import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK 다운로드 (최초 실행 시 필요)
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 1. 텍스트 전처리 함수 정의
def preprocess_text(text):
    """
    텍스트를 소문자로 변환하고, 불용어 및 구두점을 제거한 뒤 토큰화합니다.
    """
    text = text.lower()  # 소문자 변환
    tokens = word_tokenize(text)  # 토큰화
    stop_words = set(stopwords.words('english'))  # 불용어 목록 가져오기
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]  # 불용어 및 구두점 제거
    return ' '.join(tokens)  # 토큰을 다시 문자열로 결합

# 2. Query Expansion 함수 정의
def expand_query(query):
    """
    WordNet을 사용하여 쿼리의 각 단어에 대해 동의어를 추가하여 쿼리를 확장합니다.
    """
    expanded_query = []
    for word in query.split():
        synonyms = wordnet.synsets(word)  # WordNet에서 동의어 찾기
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_query.append(lemma.name())  # 동의어 추가
        expanded_query.append(word)  # 원래 단어도 유지
    return ' '.join(set(expanded_query))  # 중복 제거 후 반환

# 3. 챕터 파일 로드 및 전처리
def load_chapters(file_paths):
    """
    각 챕터 파일을 로드하고 전처리합니다.
    """
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            preprocessed_text = preprocess_text(text)  # 전처리 적용
            documents.append(preprocessed_text)
    return documents

# 4. TF-IDF 벡터화 및 코사인 유사도 계산 클래스 정의
class VectorSpaceModel:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(documents)  # TF-IDF 벡터화
    
    def process_queries(self, queries):
        """
        쿼리에 대해 Query Expansion과 전처리를 수행합니다.
        """
        expanded_queries = [expand_query(query) for query in queries]  # Query Expansion 적용
        preprocessed_queries = [preprocess_text(query) for query in expanded_queries]  # 전처리 적용
        return preprocessed_queries
    
    def calculate_similarity(self, queries):
        """
        쿼리와 문서 간 코사인 유사도를 계산합니다.
        """
        query_vectors = self.vectorizer.transform(queries)  # 쿼리를 TF-IDF 벡터로 변환
        cosine_similarities = cosine_similarity(query_vectors, self.document_vectors)  # 코사인 유사도 계산
        return cosine_similarities

# 메인 실행 코드
def main():
    # 챕터 파일 경로 설정 (예: chapter3.txt ~ chapter19.txt)
    chapter_files = [
        'chapter3.txt', 'chapter5.txt', 'chapter7.txt', 
        'chapter9.txt', 'chapter11.txt', 'chapter13.txt',
        'chapter15.txt', 'chapter17.txt', 'chapter19.txt'
    ]
    
    # 챕터 로드 및 전처리
    documents = load_chapters(chapter_files)
    
    # Vector Space Model 초기화 (챕터 기반)
    vsm = VectorSpaceModel(documents)
    
    # 질문 로드 (예: Excel 파일에서 질문 가져오기)
    question_file_path = 'Q&A_pride_and_prejudice_training.xlsx'
    questions_df = pd.read_excel(question_file_path)
    queries = questions_df['질문'].fillna("").tolist()  # 질문 컬럼 가져오기
    
    # Query Expansion 및 전처리 수행
    processed_queries = vsm.process_queries(queries)
    
    # 코사인 유사도 계산
    similarities = vsm.calculate_similarity(processed_queries)
    
    # 상위 3개 챕터 예측 결과 생성
    predictions = []
    for cos_sim in similarities:
        top_indices = (-cos_sim).argsort()[:3]  # 유사도가 높은 상위 3개 인덱스 추출
        top_chapters = [int(chapter_files[i].split('chapter')[1].split('.')[0]) for i in top_indices]
        predictions.append(top_chapters)

    # 결과 저장 및 출력
    results_df = pd.DataFrame({
        'query': queries,
        'top_1_prediction': [pred[0] for pred in predictions],
        'top_2_prediction': [pred[1] for pred in predictions],
        'top_3_prediction': [pred[2] for pred in predictions]
    })
    
    output_file_path = 'query_predictions.csv'
    results_df.to_csv(output_file_path, index=False)
    
    print(f"Results saved to {output_file_path}")

if __name__ == "__main__":
    main()