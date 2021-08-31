import csv

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from googletrans import Translator
import stop_words
from yap_server import get_lemma
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer

#manual data for testing
data = [
    ['1', 'ילדים חכמים אוהבים שוקולד'],
    ['2', 'ילדים חכמים אוהבים ילדים חכמים'],
    ['1', 'ילדות חכמות אוהבות שוקולד'],
    ['1', 'ילדים חכמים אוהבים לחקור שוקולד'],
    ['2', 'ילדות חכמות אוהבות רצים מהר'],
    ['1', 'ילדות אוהבות רצים']
]


def our_manual_tests():
    # testing with posts from the shadow. the two last ones are about the olympics, not correlating with our training set but with interesting results
    post = readify_text(
        "זה מה שמצאו מתפללים בשני בתי כנסת אתמול  בבני ברק. לבית הכנסת נזרקו גם קונדומים תמונות פורנגרפיות ותמונות של שירה בנקי זל שנרצחה במצעד הגאווה. אין מה להגיד יש הרגשה של ריפוי באויר.")
    print("psot: " + post)
    grade = grade_post(post, tfidf_vectorizer, pac)
    print(grade)
    post = readify_text(
        "פיטר פלצ'יק לאחר הזכיה אתמול במדלית ארד  באולימפיאדה היום נלחמתי לא רק בשביל עצמי ולא רק בשביל המטרות שלי והחלומות שלי, אני נלחמתי  בשביל הקבוצה, בשביל הלב שלנו, המדינה שלנו, בשביל הדגל הזה, ואני לא הולך להוריד אותו בשעות הקרובות והוא יהיה השמיכה שלי היום בלילה.")
    grade = grade_post(post, tfidf_vectorizer, pac)
    print(grade)
    post = readify_text(
        "התקווה הושמעה בטוקיו!!! תנו המון כבוד לארטיום דולגופיאט שזכה במדליית הזהב בתרגיל הקרקע באולימפיאדת טוקיו 2020. זהו ההישג הגדול ביותר לספורט הישראלי בכל הזמנים: מדליית זהב אולימפית ראשונה לישראל מאז אתונה 2004, והראשונה אי פעם באחד מענפי החשובים ביותר של המשחקים.")
    print(post)
    grade = grade_post(post, tfidf_vectorizer, pac)
    print(grade)
    # testing with a political post by miri regev
    post = readify_text(
        "דמיינו, שאתם הייתה השכנים של גדעון סער והיה לכם סכסוך נגיד על חנייה. יום למחרת גדעון סער כשר המשפטים היה מעביר חוק שפוגע בדיוק בכם באותו סכסוך על חנייה. גדעון סער מונע ממסע נקמה אישי נגד בנימין נתניהו, יש כאן ניגוד עניינים ברור והוא לא יכול להתעסק בשום הצעת חוק הקשורה לנתניהו. ")
    grade = grade_post(post, tfidf_vectorizer, pac)
    print("Miri Regev's post grade: " + str(grade))
    # merav michaeli post
    post = readify_text(
        "הבריאות שלנו היא מעל הכל. סיכמתי עם ראש הממשלה נפתלי בנט - Naftali Bennett ועם משרד האוצר על גיוס של 400 פקחים אשר יאכפו את עטיית המסיכות בתחבורה הציבורית כדי למנוע הדבקה. המלחמה בקורונה היא לטובת כולנו - אל תזלזלו והקפידו על עטיית מסיכה. ")
    grade = grade_post(post, tfidf_vectorizer, pac)
    print(grade)

"""
    Given a post, a _ , and a classifier,
    return a grade between 0-5 indicating the level of fake
"""
def grade_post (post, machine, classifer):
    tfidf_test = machine.transform([post])
    y_pred = classifer.predict(tfidf_test)
    return y_pred

"""
    Remove punctuation
"""
def clean_txt (txt):
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in txt:
        if ele in punc:
            txt = txt.replace(ele, "")
    return txt

"""
    Remove frequent and unimportant words
"""
def remove_stopwords(txt):
    stopwords = stop_words.stop_words
    tokens = txt.split(" ")
    resultwords = [word for word in tokens if word not in stopwords]
    result = ' '.join(resultwords)
    return result

def readify_text(txt):
    txt = clean_txt(txt)
    txt = remove_stopwords(txt)
    #TODO stem? https://towardsdatascience.com/getting-your-text-data-ready-for-your-natural-language-processing-journey-744d52912867
    return txt

def our_svm(tfidf_train, label_train, tfidf_valid, label_valid):
    # create a svm classifier
    svm_clf = svm.SVC(kernel='linear')

    #train
    svm_clf.fit(tfidf_train, label_train)

    #predict the labels for the text validation data
    label_prediction = svm_clf.predict(tfidf_valid)

    #check model accuracy
    print("Accuracy:", metrics.accuracy_score(label_valid, label_prediction))

    print(confusion_matrix(label_valid, label_prediction))


'''
my custom csv file for understanding whats going on
'''

def my_csv_writer():
    header = ['label', 'text']
    with open('my_csv_file.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    df = pd.read_csv('my_csv_file.csv')
    df = df.drop_duplicates()
    df['lemmatized_text'] = df.apply(lambda row: get_lemma(row['text']), axis=1)
    df.to_csv('my_csv_file_lemmatized.csv', encoding='utf-8', index=False)


"""
    This function reads the raw file mashrokit.csv
    Then it applies modification for the file, making a new file
    with text that is ready for the learning part.
    It should be called after we scrap the mashrokit website and enter new data
"""
def add_lemmas():
    df = pd.read_csv('mashrokit.csv')
    df = df.drop_duplicates()
    # df = df[:5]
    #create a new column which will contain the texts after lemmatification
    df['lemmatized_text'] = df.apply(lambda row: get_lemma(row['text']), axis=1)
    df.to_csv('mashrokit_with_lemmas.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    # my_csv_writer()
    # df = pd.read_csv('my_csv_file_lemmatized.csv')

    # *************** TRYING COVID ***************
    df_true = pd.read_csv('trueNews.csv')
    df_false = pd.read_csv('fakeNews.csv')
    # df_false['our_labels'] = df_false.apply(lambda col: col['Poynter_Label'].upper(), axis=1)
    df_true.dropna(inplace=True)
    df_false.dropna(inplace=True)
    frames = [df_true, df_false]
    df = pd.concat(frames, join='inner')

    # labels = df_false['our_labels']
    labels = df['Binary Label']
    # labels += df_true['Label']
    text_train, text_valid, label_train, label_valid = train_test_split(df['Text'], labels, test_size=0.2,
                                                                        random_state=109)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', ngram_range=(1, 1), norm=None)
    tfidf_train = tfidf_vectorizer.fit_transform(text_train)
    tfidf_valid = tfidf_vectorizer.transform(text_valid)
    our_svm(tfidf_train, label_train, tfidf_valid, label_valid)



    # *************** FINISH **********************
    #
    #
    # # add lemmas - takes a long time, add when mashrokit.csv changes
    # # add_lemmas()
    # df = pd.read_csv('mashrokit_with_lemmas.csv')
    # # df.dropna() #remove rows with none
    # # df = df.loc[df['label'] != 0] #remove rows with label = 0
    # print('finished lemmas')
    #
    # # # DataFlair - Get the labels
    # labels = df.label
    # # validation
    # text_train, text_valid, label_train, label_valid = train_test_split(df['lemmatized_text'], labels, test_size=0.2, random_state=109)
    #
    # #create vectorizer to work with numbers instead of text TODO clean text
    # tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words.stop_words, strip_accents='unicode', ngram_range=(1, 1), norm=None)
    # #in fit_transform, count_vocab: creates a vocabulary which is a dictionary
    # #consisting of all n-grams. also creates a matrix of vocabulary and num of occurences for each. called X
    # tfidf_train = tfidf_vectorizer.fit_transform(text_train)
    # tfidf_valid = tfidf_vectorizer.transform(text_valid)
    #
    # #print vectorizer contents
    # feature_names = tfidf_vectorizer.get_feature_names()
    # # corpus_index = [n for n in range(len(data))]
    # # rows, cols = tfidf_train.nonzero()
    # # for row, col in zip(rows, cols):
    # #     print((feature_names[col], corpus_index[row]), tfidf_train[row, col])
    # # print(tfidf_train)
    # df_for_print = pd.DataFrame(tfidf_train.T.todense(), index=feature_names, columns=text_train.index)
    # print(df_for_print)
    # # hashing_vectorizer = HashingVectorizer(stop_words=stop_words.stop_words, strip_accents='unicode', ngram_range=(1, 3), norm='l1')
    # # hashing_train = hashing_vectorizer.fit_transform(text_train)
    # # hashing_valid = hashing_vectorizer.transform(text_valid)
    #
    # #do svm
    # our_svm(tfidf_train, label_train, tfidf_valid, label_valid)
    # # our_svm(hashing_train, label_train, hashing_valid, label_valid)




"""
optional classifiers to add:
1. random forest classifier (from sklearn)
2. Multinomial Naive Bayes Algorithm (need to read more about it, we're not sure how it fits in). (from: https://www.analyticsvidhya.com/blog/2021/06/build-your-own-fake-news-classifier-with-nlp/)
3. we have 5 different classifier examples in the article:
https://www.researchgate.net/profile/Marten-Risius/publication/326405790_Automatic_Detection_of_Fake_News_on_Social_Media_Platforms/links/5b7df935a6fdcc5f8b5de39c/Automatic-Detection-of-Fake-News-on-Social-Media-Platforms.pdf
(page 11)
we want to create different functions for each classifier, and somehow work with the different results to get a more accurate analysis
"""