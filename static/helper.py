import pickle
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

with open(MODEL_DIR / "supported_vector_machine.pkl", "rb") as file:
    model = pickle.load(file)

with open(MODEL_DIR / "corpora" / "stopwords" / "english", "r") as file:
    stop_words = file.read().splitlines()

vocab = pd.read_csv(MODEL_DIR / "vocabulary.txt", header=None)
tokens = vocab[0].tolist()
stemmer = PorterStemmer()


def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    return text


def preprocessing(text):
    data = pd.DataFrame([text], columns=["tweet"])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data["tweet"].apply(
        lambda x: " ".join(re.sub(r"^http:?:\/\/.*[\r\n]*", "", x, flags=re.MULTILINE) for x in x.split())
    )
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(remove_punctuation(x) for x in x.split()))
    data["tweet"] = data["tweet"].str.replace(r"\d+", "", regex=True)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(stemmer.stem(x) for x in x.split()))
    return data["tweet"]


def vectorizer(ds):
    vectorized = []

    for sentence in ds:
        sentence_vector = np.zeros(len(tokens))

        for i, token in enumerate(tokens):
            if token in sentence.split():
                sentence_vector[i] = 1

        vectorized.append(sentence_vector)

    return np.array(vectorized, dtype=np.float32)


def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    return "negative" if prediction[0] == 1 else "positive"
