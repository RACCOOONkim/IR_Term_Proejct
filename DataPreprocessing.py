# 텍스트 전처리 코드 정리한 내용입니다. 작성자: 전소현

# 텍스트 전처리 (preprocess_text 함수)
# - 소문자 변환
# - 특수문자 제거 (단어, 공백, 마침표, 쉼표, 물음표 제외)
# - 토큰화
# - 불용어 제거
# - 어간 추출

# 책 내용 로드 및 전처리 (load_and_preprocess_book 함수)
# - UTF-8 인코딩으로 텍스트 파일 읽음
# - 책 시작 부분 찾음
# - 챕터 분리
# - 과제에 사용할 챕터 추출 (3, 5, 7, 9, 11, 13, 15, 17, 19장)
# - 각 챕터 개별 전처리

# 로마 숫자 변환 (roman_to_integer 함수)
# - 로마 숫자를 일반 숫자로 변환 (Chapter I, II, V 방식으로 되어있음)

# 사용된 자연어 처리 기법

# 1. Tokenization: 텍스트를 개별 단어로 분리
# 2. Stop words removal: 의미 없는 일반적인 단어 제거
# 3. Stemming: 단어의 어간을 추출하여 유사 의미 단어 통일
# 4. Normalization: 소문자 변환 및 특수문자 제거

# ----------

# 파이썬 코드

import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import os
import math

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    
    text = re.sub(r'[^\w\s.,?]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

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
        
        text = text[start_index:]
        
        chapters = []
        chapter_numbers = [3, 5, 7, 9, 11, 13, 15, 17, 19]
        
        chapter_pattern = r"Chapter\s+(\d+|[IVXLCDM]+)"
        chapter_matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))
        
        for i, match in enumerate(chapter_matches):
            chapter_num = match.group(1)
            if chapter_num.isdigit():
                chapter_num = int(chapter_num)
            else:
                chapter_num = roman_to_integer(chapter_num)
            
            if chapter_num in chapter_numbers:
                if i+1 < len(chapter_matches):
                    chapter_text = text[match.start():chapter_matches[i+1].start()]
                else:
                    chapter_text = text[match.start():]
                
                chapters.append(chapter_text)
                print(f"Found Chapter {chapter_num}. Length: {len(chapter_text)} characters")
        
        preprocessed_chapters = [preprocess_text(chapter) for chapter in chapters]
        
        print(f"Total chapters found: {len(chapters)}")
        
        return preprocessed_chapters, chapter_numbers
    except Exception as e:
        print(f"Error in load_and_preprocess_book: {str(e)}")
        return [], []

def roman_to_integer(roman):
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    integer = 0
    for i in range(len(roman)):
        if i > 0 and roman_values[roman[i]] > roman_values[roman[i - 1]]:
            integer += roman_values[roman[i]] - 2 * roman_values[roman[i - 1]]
        else:
            integer += roman_values[roman[i]]
    return integer

def main():
    book_path = 'pandp12p.txt'
    chapters, chapter_numbers = load_and_preprocess_book(book_path)
    
    print(f"Number of chapters loaded: {len(chapters)}")
    print(f"Chapter numbers: {chapter_numbers}")

if __name__ == "__main__":
    main()