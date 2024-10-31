# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from selenium import webdriver
from bs4 import BeautifulSoup as bs

# NLTK 리소스 다운로드
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
    
    stemmer = PorterStemmer()  # Stemmer 초기화
    tokens = [stemmer.stem(token) for token in tokens]  # Stemming 적용
    
    return ' '.join(tokens)

# 웹에서 책을 로드하고 전처리하는 함수 (구텐베르크에서 크롤링)
def load_and_preprocess_book_from_web():
    try:
        url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
        
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        soup = bs(driver.page_source, 'html.parser')
        contents = soup.find('body').string
        
        driver.quit()  # 드라이버 종료
        
        print(f"Successfully fetched text from the web. Total characters: {len(contents)}")
        
        # 텍스트 전처리 (본문 및 챕터 분리)
        contents_head_cut = contents.split('Chapter I.]')[1].lstrip()
        
        tail = '''                            [Illustration:
                                  THE END ]'''
        contents_tail_cut = contents_head_cut.split(tail)[0].rstrip()
        
        # 챕터별로 분리 (CHAPTER 기준)
        contents_chapters = contents_tail_cut.split('CHAPTER')
        
        contents_chapters_cut = []
        contents_chapters_cut.append(contents_chapters[0])
        
        for i in range(1, len(contents_chapters)):
            contents_chapters_cut.append(contents_chapters[i][7:])
        
        # 불필요한 텍스트 정리
        def clean_chapters(chapter):
            new_content = chapter.replace('\n', ' ').replace('[Illustration]', '')
            pattern1 = r'\[Illustration.*?\]'
            new_content = re.sub(pattern1, '', new_content)
            new_content = new_content.replace(' ] ', '').replace('  ', ' ')
            return new_content
        
        contents_chapters_cleaned = [clean_chapters(chapter) for chapter in contents_chapters_cut]
        
        chapters_target = [3, 5, 7, 9, 11, 13, 15, 17, 19]  # 선택할 챕터 목록
        
        target_txt = [contents_chapters_cleaned[target - 1] for target in chapters_target]
        
        preprocessed_chapters = [preprocess_text(chapter) for chapter in target_txt]
        
        print(f"Total chapters found and preprocessed: {len(preprocessed_chapters)}")
        
        return preprocessed_chapters, chapters_target
    
    except Exception as e:
        print(f"Error in load_and_preprocess_book_from_web: {str(e)}")
        return [], []

# Vector Space Model 클래스 (질문과 챕터 유사도 계산)
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
            'Top N Chapters': self.top_n_chapters
        })
        return top_n_df

def main():
    # 책을 웹에서 로드하고 전처리
    chapters, chapter_numbers = load_and_preprocess_book_from_web()
    
    if chapters:
        vsm = VectorSpaceModel(chapters, chapter_numbers)
        
        question_file_path = 'Q&A_pride and prejudice_training.xlsx'
        
        vsm.load_questions(question_file_path)
        vsm.vectorize_queries()
        vsm.calculate_similarity()
        vsm.get_top_n_chapters(n=3)
        
        results = vsm.output_results()
        
        print(results)
        
    # # 추론된 결과 (상위 3개의 챕터 번호)
    # predictions = [
    #     [11, 3, 7], [17, 15, 3], [7, 15, 5], [7, 17, 11], [3, 17, 9],
    #     [17, 3, 9], [3, 17, 9], [7, 9, 17], [13, 19, 15], [13, 7, 17],
    #     [3, 15, 7], [19, 17, 7], [3, 7, 5], [15, 19, 13], [17, 9, 7],
    #     [7, 15, 17], [17, 7, 11], [5, 7, 13], [17, 3, 9], [15, 3, 17],
    #     [7, 9, 3], [13, 9, 7], [11, 7, 5], [17, 7, 9], [7, 9, 15],
    #     [17, 15, 3], [3, 15, 17]
    # ]

    # # 정답지 (정답이 하나 또는 두 개일 수 있음)
    # answers = [
    #     [11], [15], [7], [11], [3],
    #     [17], [5,3], [7], [13,15], [7],
    #     [3], [19], [3], [13], [9],
    #     [9], [5], [5], [17], [15],
    #     [13], [7], [11], [17], [7],
    #     [19], [3]
    # ]

    # # 정답률 계산
    # correct_count = sum(any(pred in answers[i] for pred in predictions[i]) for i in range(len(predictions)))
    # total_questions = len(predictions)
    # accuracy = correct_count / total_questions * 100

    # print(f"정답률: {accuracy:.2f}%")

if __name__ == "__main__":
    main()