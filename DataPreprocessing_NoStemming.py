# Stemming 대신 lLemmatization 사용.

import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import os
import math

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng')

# 단어의 품사 태그를 WordNet 형식으로 변환하는 함수
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()  # 품사 태그 추출
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # 명사가 기본값

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

def load_and_preprocess_book(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()  # 텍스트 파일 읽기
        
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
    book_path = 'pandp12p.txt'  # 책 파일 경로
    chapters, chapter_numbers = load_and_preprocess_book(book_path)  # 책 로드 및 전처리
    print(f"Number of chapters loaded: {len(chapters)}")
    print(f"Chapter numbers: {chapter_numbers}")

if __name__ == "__main__":
    main()