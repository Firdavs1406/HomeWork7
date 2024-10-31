import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from collections.abc import Iterable
from gensim.models import FastText
import numpy as np


def load_and_preprocess_data():
    data = load_dataset('ucirvine/sms_spam')
    data.set_format(type='pandas')
    df = data['train'].to_pandas()
    df['sms'] = df['sms'].apply(lambda x: x.lower())
    df['sms'] = df['sms'].apply(word_tokenize)

    df.to_csv('data/processed_data.csv', index=False)
    return df


def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df.sms.values, df.label.values, test_size=0.3, random_state=42
    )

    train_df = pd.DataFrame({'sms': X_train, 'label': y_train})
    test_df = pd.DataFrame({'sms': X_test, 'label': y_test})
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    return X_train, X_test, y_train, y_test


def get_text_corpus(texts: Iterable[list]) -> list:
    corpus = [word for text in texts for word in text]
    return list(set(corpus))


def build_emb_dict(corpus: list, model) -> dict:
    emb_dict = {}
    for word in corpus:
        emb_dict[word] = model.wv[word]
    return emb_dict


def prepare_embeddings(X_train, X_test, emb_dim):
    corpus = get_text_corpus(X_train)
    ft = FastText(vector_size=emb_dim, window=3, min_count=1)
    ft.build_vocab(corpus_iterable=X_train)
    ft.train(corpus_iterable=X_train, total_examples=len(X_train), epochs=10)
    emb_dict = build_emb_dict(corpus, ft)
    X_train_emb = convert_to_emb(X_train, emb_dict)
    X_test_emb = convert_to_emb(X_test, emb_dict)
    return X_train_emb, X_test_emb, emb_dict


def convert_to_emb(X: np.ndarray, emb_dict: dict) -> np.ndarray:
    default_dim = len(next(iter(emb_dict.values())))
    return [
        np.array([emb_dict.get(word, np.zeros(default_dim)) for word in sample])
        for sample in X
    ]
