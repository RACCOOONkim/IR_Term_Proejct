import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# 단어의 품사 태그를 WordNet 형식으로 변환하는 함수
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ, "N": nltk.corpus.wordnet.NOUN, "V": nltk.corpus.wordnet.VERB, "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

# 텍스트 전처리 함수
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    
    text = re.sub(r'[^\w\s.,?]', '', text.lower())  # 소문자 변환 및 특수문자 제거
    tokens = word_tokenize(text)  # 토큰화
    stop_words = set(stopwords.words('english'))  # 불용어 설정
    tokens = [token for token in tokens if token not in stop_words]  # 불용어 제거
    
    lemmatizer = WordNetLemmatizer()  # Lemmatizer 초기화
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]  # Lemmatization 적용
    
    return ' '.join(tokens)

# 책을 로드하고 전처리하는 함수
def load_and_preprocess_book(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        print(f"Successfully read file. Total characters: {len(text)}")
        
        start_index = text.lower().find("chapter 1")
        if start_index == -1:
            start_index = text.find("It is a truth universally acknowledged")
        
        if start_index == -1:
            print("Could not find the start of the book content.")
            return [], []
        
        text = text[start_index:]  # 책 시작 부분부터 잘라내기
        
        chapters = []
        chapter_numbers = [3, 5, 7, 9, 11, 13, 15, 17, 19]  # 선택할 챕터 목록
        
        # 챕터 찾기
        chapter_pattern = r"Chapter\s+(\d+|[IVXLCDM]+)"  # 정규식으로 챕터 찾기
        chapter_matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))
        
        for i, match in enumerate(chapter_matches):
            chapter_num = match.group(1)
            if chapter_num.isdigit():
                chapter_num = int(chapter_num)
            else:
                chapter_num = roman_to_integer(chapter_num)  # 로마 숫자 변환
                
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

# 로마 숫자 변환 함수
def roman_to_integer(roman):
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    integer = 0
    for i in range(len(roman)):
        if i > 0 and roman_values[roman[i]] > roman_values[roman[i - 1]]:
            integer += roman_values[roman[i]] - 2 * roman_values[roman[i - 1]]
        else:
            integer += roman_values[roman[i]]
    return integer

# Vector Space Model 클래스
class VectorSpaceModel:
    def __init__(self, preprocessed_chapters):
        """
        Initialize the Vector Space Model with preprocessed chapters.
        """
        self.preprocessed_chapters = preprocessed_chapters
        self.vectorizer = TfidfVectorizer()
        self.chapter_vectors = self.vectorizer.fit_transform(preprocessed_chapters)
    
    def load_questions(self, file_path):
        """
        Load the questions from an Excel file.
        """
        qa_data = pd.read_excel(file_path)
        self.questions = qa_data['질문'].fillna("").tolist()
    
    def vectorize_queries(self):
        """
        Vectorize the queries (questions) using the same vectorizer as the chapters.
        """
        self.query_vectors = self.vectorizer.transform(self.questions)
    
    def calculate_similarity(self):
        """
        Calculate cosine similarity between queries and chapters.
        """
        self.cosine_sim = cosine_similarity(self.query_vectors, self.chapter_vectors)
    
    def get_top_n_chapters(self, n=3):
        """
        Get the top N chapters for each query based on cosine similarity.
        """
        self.top_n_chapters = [(-cos_sim).argsort()[:n] + 1 for cos_sim in self.cosine_sim]  # +1 for chapter numbering
    
    def output_results(self):
        """
        Output the results as a DataFrame showing the top N chapters for each query.
        """
        top_n_df = pd.DataFrame({
            'Question': self.questions,
            'Top N Chapters': self.top_n_chapters
        })
        return top_n_df

def main():
    # 책 파일 경로 설정
    book_path = 'pandp12p.txt'
    
    # 책을 로드하고 전처리
    chapters, chapter_numbers = load_and_preprocess_book(book_path)
    
    if chapters:
        # Vector Space Model 초기화
        vsm = VectorSpaceModel(chapters)
        
        # 질문 파일 경로 설정
        question_file_path = 'Q&A_pride and prejudice_training 문제only(2024).xlsx'
        
        # 질문 로드
        vsm.load_questions(question_file_path)
        
        # 질문 벡터화
        vsm.vectorize_queries()
        
        # 코사인 유사도 계산
        vsm.calculate_similarity()
        
        # 상위 3개의 챕터 가져오기
        vsm.get_top_n_chapters(n=3)
        
        # 결과 출력
        results = vsm.output_results()
        print(results)
        
        # 결과를 엑셀 파일로 저장
        output_file = 'question_answers_output.xlsx'
        results.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()