import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import stop_words

def remove_mystopwords(sentence):
    my_stopwords = stop_words.stop_words
    sentence = sentence.replace('\n', ' ')
    sentence = sentence.strip(':')
    sentence = sentence.strip(',')
    sentence = sentence.strip('.')
    sentence = sentence.strip('\\')
    tokens = sentence.split(" ")
    print(tokens)
    tokens_filtered = [word for word in tokens if word not in my_stopwords]
    return (" ").join(tokens_filtered)

if __name__ == '__main__':
    # Read the data
    df = pd.read_csv('data.csv')

    # Get shape and head
    df.shape
    df.head()
    # DataFlair - Get the labels
    labels = df.label
    labels.head()
    # DataFlair - Split the dataset
    new_text = []
    # date, text, author, label, index
    # texts = df['text']
    i = 0
    for text, label in zip(df['text'], df['label']):
        # print(text)
        new_text.insert(i, (label, remove_mystopwords(str(text))))
        i += 1
        # new_text.insert(i,remove_mystopwords(str(df['text'][i])))

        # print(text)
    x_train, x_test, y_train, y_test = train_test_split(new_text[1], new_text[0], test_size=0.2, random_state=7)
    # DataFlair - Initialize a TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.1)

    # DataFlair - Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    # DataFlair - Initialize a PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # DataFlair - Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')
    # DataFlair - Build confusion matrix
    confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])