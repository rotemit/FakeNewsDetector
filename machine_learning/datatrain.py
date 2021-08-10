import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from googletrans import Translator
import stop_words
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer

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

if __name__ == '__main__':
    # Read the data
    df = pd.read_csv('mashrokit.csv')
    # df = list(dict.fromkeys(df))
    df = df.drop_duplicates()
    print(df.shape)
    # # Get shape and head
    # print(df.shape)
    # print(df.head(5))
    # # DataFlair - Get the labels
    labels = df.label
    # print(labels.head(5))
    # validation
    text_train, text_valid, label_train, label_valid = train_test_split(df['text'], labels, test_size=0.2, random_state=109)
    # print(label_train.count)
    #readify texts
    # text_train = [(index, readify_text(sentence)) for (index, sentence) in text_train]
    # print(text_train.shape)
    # text_valid = [(index, readify_text(sentence)) for index, sentence in text_valid]
    #create vectorizer to work with numbers instead of text
    # tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words.stop_words, strip_accents='unicode', ngram_range=(2, 5))
    # tfidf_train = tfidf_vectorizer.fit_transform(text_train)
    # tfidf_valid = tfidf_vectorizer.transform(text_valid)

    hashing_vectorizer = HashingVectorizer(stop_words=stop_words.stop_words, strip_accents='unicode', ngram_range=(1, 3), norm='l1')
    hashing_train = hashing_vectorizer.fit_transform(text_train)
    hashing_valid = hashing_vectorizer.transform(text_valid)

    #do svm
    # our_svm(tfidf_train, label_train, tfidf_valid, label_valid)
    our_svm(hashing_train, label_train, hashing_valid, label_valid)


    # # DataFlair - Split the dataset WE DONT WANT THIS BECAUSE WERE ONLY LOOKING FOR TRAINING DATA
    # # x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    # # DataFlair - Initialize a TfidfVectorizer
    # tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
    # x_train = [readify_text(sentence) for sentence in df["text"]]
    # y_train = labels
    # # DataFlair - Fit and transform train set, transform test set
    # tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    # # tfidf_test = tfidf_vectorizer.transform(x_test)
    # # DataFlair - Initialize a PassiveAggressiveClassifier
    # pac = PassiveAggressiveClassifier(max_iter=50)
    # pac.fit(tfidf_train, y_train)



"""
optional classifiers to add:
1. random forest classifier (from sklearn)
2. Multinomial Naive Bayes Algorithm (need to read more about it, we're not sure how it fits in). (from: https://www.analyticsvidhya.com/blog/2021/06/build-your-own-fake-news-classifier-with-nlp/)
3. we have 5 different classifier examples in the article:
https://www.researchgate.net/profile/Marten-Risius/publication/326405790_Automatic_Detection_of_Fake_News_on_Social_Media_Platforms/links/5b7df935a6fdcc5f8b5de39c/Automatic-Detection-of-Fake-News-on-Social-Media-Platforms.pdf
(page 11)
we want to create different functions for each classifier, and somehow work with the different results to get a more accurate analysis
"""