import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer

import config

# --- Tải tài nguyên NLTK ---
def download_nltk_resources():
    resources = {'corpora/wordnet.zip': 'wordnet', 'tokenizers/punkt': 'punkt', 'corpora/stopwords': 'stopwords'}
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Đang tải tài nguyên NLTK: {name}...")
            nltk.download(name, quiet=True)
    print("Tất cả tài nguyên NLTK cần thiết đã sẵn sàng.")

# --- Các hàm làm sạch và xử lý văn bản ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    marks_and_digits = r'''!()-[]{};?@#$%:'"\\,|./^&;*_0123456789'''
    text = ''.join(char for char in text if char not in marks_and_digits)
    unwanted_phrases = ['url', 'privacy policy', 'disclaimer', 'copyright policy']
    for phrase in unwanted_phrases:
        text = text.replace(phrase, '')
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_and_process(text, stop_words, processor_func, processor_type):
    tokens = text.split()
    processed_tokens = []
    for word in tokens:
        if word not in stop_words and len(word) > 2:
            if processor_type == 'stem':
                processed_tokens.append(processor_func(word))
            elif processor_type == 'lem':
                processed_tokens.append(processor_func(word, pos='v'))
    return processed_tokens

# --- Hàm chính để tải và tiền xử lý dữ liệu ---
def load_and_preprocess_data():
    download_nltk_resources()
    
    try:
        df = pd.read_csv(config.DATA_PATH, on_bad_lines="skip", engine='python')
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {config.DATA_PATH}")

    df = df.fillna('')
    df['case_text_sum'] = df['case_title'] + " " + df['case_text']
    df['clean_text'] = df['case_text_sum'].apply(clean_text)
    
    le = LabelEncoder()
    df['case_outcome_num'] = le.fit_transform(df['case_outcome'])
    num_classes = df['case_outcome_num'].nunique()
    
    stop_words_list = nltk_stopwords.words('english')
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    
    print("Đang xử lý Stemming...")
    df['tokens_stm'] = df['clean_text'].apply(lambda x: tokenize_and_process(x, stop_words_list, porter_stemmer.stem, 'stem'))
    df['text_stm_joined'] = df['tokens_stm'].apply(' '.join)
    
    print("Đang xử lý Lemmatization...")
    df['tokens_lem'] = df['clean_text'].apply(lambda x: tokenize_and_process(x, stop_words_list, wordnet_lemmatizer.lemmatize, 'lem'))
    df['text_lem_joined'] = df['tokens_lem'].apply(' '.join)
    
    print("Đã hoàn tất tiền xử lý.")
    return df, le, num_classes

def get_ml_datasets(df):
    y = df['case_outcome_num']
    
    # Dữ liệu cho LR (Stemming)
    X_stm_text = df['text_stm_joined']
    train_stm_text, test_stm_text, y_train_stm, y_test_stm = train_test_split(
        X_stm_text, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Vectorize cho LR
    tfidf_vec_stm = TfidfVectorizer(max_features=config.MAX_FEATURES_ML)
    X_train_stm_tfidf = tfidf_vec_stm.fit_transform(train_stm_text)
    X_test_stm_tfidf = tfidf_vec_stm.transform(test_stm_text)
    
    # Dữ liệu cho LSVC (Lemmatization)
    X_lem_text = df['text_lem_joined']
    train_lem_text, test_lem_text, y_train_lem, y_test_lem = train_test_split(
        X_lem_text, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Vectorize cho LSVC
    tfidf_vec_lem = TfidfVectorizer(max_features=config.MAX_FEATURES_ML)
    X_train_lem_tfidf = tfidf_vec_lem.fit_transform(train_lem_text)
    X_test_lem_tfidf = tfidf_vec_lem.transform(test_lem_text)
    
    return (X_train_stm_tfidf, X_test_stm_tfidf, y_train_stm, y_test_stm), \
           (X_train_lem_tfidf, X_test_lem_tfidf, y_train_lem, y_test_lem)

def get_cnn_datasets(df):
    # CNN dùng Lemmatization
    y = df['case_outcome_num']
    X_lem_tokens = df['tokens_lem']
    
    train_lem_tokens, test_lem_tokens, y_train_lem, y_test_lem = train_test_split(
        X_lem_tokens, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Huấn luyện FastText trên tập train
    print("Đang huấn luyện FastText model...")
    ft_model = FastText(sentences=train_lem_tokens, vector_size=config.W2V_SIZE, window=config.W2V_WINDOW, 
                        min_count=config.W2V_MIN_COUNT, workers=config.W2V_WORKERS, sg=1)
    
    return train_lem_tokens, test_lem_tokens, y_train_lem, y_test_lem, ft_model