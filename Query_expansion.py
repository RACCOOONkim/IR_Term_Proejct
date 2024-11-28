import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# 단어의 품사 태그를 WordNet 형식으로 변환하는 함수
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ, "N": nltk.corpus.wordnet.NOUN, "V": nltk.corpus.wordnet.VERB, "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

# Query Expansion 함수: 동의어를 활용하여 쿼리 확장
def expand_query(query):
    expanded_query = []
    for word in query.split():
        # WordNet에서 동의어 찾기
        synonyms = wordnet.synsets(word)
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_query.append(lemma.name())  # 동의어 추가
        expanded_query.append(word)  # 원래 단어도 유지
    return ' '.join(set(expanded_query))  # 중복 제거 후 반환

# 텍스트 전처리 함수
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    
    text = re.sub(r'[^\w\s.,?]', '', text.lower())  # 소문자 변환 및 특수문자 제거
    tokens = word_tokenize(text)  # 토큰화
    stop_words = set(stopwords.words('english'))  # 불용어 설정
    tokens = [token for token in tokens if token not in stop_words]  # 불용어 제거
    
    stemmer = PorterStemmer()  # Stemmer 초기화
    tokens = [stemmer.stem(token) for token in tokens]  # Stemming 적용
    
    return ' '.join(tokens)

# 책을 로드하고 전처리하는 함수
def load_and_preprocess_book(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        print(f"Successfully read file. Total characters: {len(text)}")
        
        chapters = []
        chapter_numbers = [3, 5, 7, 9, 11, 13, 15, 17, 19]  # 선택할 챕터 목록
        
        # 챕터 찾기
        chapter_pattern = r"Chapter\s+(\d+)"  # 정규식으로 챕터 찾기 (숫자만)
        chapter_matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))
        
        for i, match in enumerate(chapter_matches):
            chapter_num = int(match.group(1))
            if chapter_num in chapter_numbers:  # 원하는 챕터일 경우 추출
                if i+1 < len(chapter_matches):
                    chapter_text = text[match.start():chapter_matches[i+1].start()]
                else:
                    chapter_text = text[match.start():]
                
                chapters.append(chapter_text)
                print(f"Found Chapter {chapter_num}. Length: {len(chapter_text)} characters")
        
        preprocessed_chapters = [preprocess_text(chapter) for chapter in chapters]  # 각 챕터 전처리
        
        print(f"Total chapters found: {len(chapters)}")
        
        return preprocessed_chapters, chapter_numbers  # 전처리된 챕터 및 번호 반환
    except Exception as e:
        print(f"Error in load_and_preprocess_book: {str(e)}")
        return [], []

# Vector Space Model 클래스 수정 (Query Expansion 추가)
class VectorSpaceModel:
    def __init__(self, preprocessed_chapters, chapter_numbers):
        self.preprocessed_chapters = preprocessed_chapters
        self.chapter_numbers = chapter_numbers
        self.vectorizer = TfidfVectorizer()
        self.chapter_vectors = self.vectorizer.fit_transform(preprocessed_chapters)
    
    def load_questions(self, file_path):
        """
        Load the questions from an Excel file.
        """
        qa_data = pd.read_excel(file_path)
        self.questions = qa_data['질문'].fillna("").tolist()
    
    def expand_and_preprocess_queries(self):
        """
        Expand and preprocess the queries using Query Expansion.
        """
        expanded_queries = [expand_query(q) for q in self.questions]  # Query Expansion 적용
        preprocessed_queries = [preprocess_text(q) for q in expanded_queries]  # 전처리 적용
        self.expanded_questions = preprocessed_queries
    
    def vectorize_queries(self):
        """
        Vectorize the queries (questions) using the same vectorizer as the chapters.
        """
        self.query_vectors = self.vectorizer.transform(self.expanded_questions)
    
    def calculate_similarity(self):
        """
        Calculate cosine similarity between queries and chapters.
        """
        self.cosine_sim = cosine_similarity(self.query_vectors, self.chapter_vectors)
    
    def get_top_n_chapters(self, n=3):
            self.top_n_chapters = []
            for cos_sim in self.cosine_sim:
                top_indices = (-cos_sim).argsort()[:n]
                top_chapters = [self.chapter_numbers[i] for i in top_indices]
                self.top_n_chapters.append(top_chapters)
    
    def output_results(self):
        """
        Output the results as a DataFrame showing the top N chapters for each query.
        """
        top_n_df = pd.DataFrame({
            'Question': self.questions,
            'Expanded Question': self.expanded_questions,
            'Top N Chapters': self.top_n_chapters
        })
        return top_n_df

def main():
    book_path = 'cleaned_pandp12p.txt'
    chapters, chapter_numbers = load_and_preprocess_book(book_path)
    
    if chapters:
        vsm = VectorSpaceModel(chapters, chapter_numbers)
        
        question_file_path = 'Q&A_pride and prejudice_training.xlsx'
        
        vsm.load_questions(question_file_path)
        
        vsm.expand_and_preprocess_queries()  # Query Expansion 및 전처리 추가 단계
        
        vsm.vectorize_queries()
        
        vsm.calculate_similarity()
        
        vsm.get_top_n_chapters(n=3)
        
        results = vsm.output_results()
        
        print(results)
        
if __name__ == "__main__":
    main()