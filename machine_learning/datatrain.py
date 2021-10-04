import csv
#testtt
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from machine_learning.stop_words import stop_words
# from yap_server import get_lemma
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import linear_model
from deep_translator import GoogleTranslator #pip installed
# from machine_learning.heb_data_collector import get_group_posts
from sklearn.linear_model import LogisticRegressionCV
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
# import nltk
# nltk.download('wordnet')

def grade_single_post(post,svm_model, vectorizer):
    #load trained model and fitted vectorizer
    svm_model = joblib.load(svm_model)
    vectorizer = joblib.load(vectorizer)
    #translate post to english, regardless of source language
    translator = GoogleTranslator()
    translated_post = translator.translate(post)
    readied_post = readify_text(translated_post)
    #vectorize and predict fakeness
    vectorized_post = vectorizer.transform([readied_post])
    y_pred = svm_model.predict(vectorized_post)
    #print result
    print("post:\n" + post + "\ntranslated post:\n" + translated_post + "\nreadied post:\n"+readied_post+"\ngrade: " + str(y_pred))
    return y_pred

'''
    manual tests for svm
'''
def our_svm_tests(tfidf_vectorizer, svm_classifier):
    posts = pd.read_csv('heb_posts.csv')
    for post in posts:
        en_post = GoogleTranslator(source='he', target='en').translate(post)
        grade = grade_post(en_post, tfidf_vectorizer, svm_classifier)
        print('Post: '+en_post+'\nGrade: '+str(grade)+'\n')

"""
    Given a post, a _ , and a classifier,
    return a binary grade indicating the level of fake
"""
def grade_post (post, machine, classifer):
    tfidf_test = machine.transform([post])
    y_pred = classifer.predict(tfidf_test)
    return y_pred

"""
    Remove punctuation
"""
def clean_txt (txt):
    #make lowercase
    txt = txt.lower()
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
    stopwords = stop_words
    tokens = txt.split(" ")
    resultwords = [word for word in tokens if word not in stopwords]
    result = ' '.join(resultwords)
    return result

def our_stemmer(txt):
    # ps = PorterStemmer()
    # toreturn = ' '.join(ps.stem(word) for word in txt.split())
    # return toreturn
    ls = LancasterStemmer()
    toreturn = ' '.join(ls.stem(word) for word in txt.split())
    return toreturn

def our_lemmatizer(txt):
    wordnet_lemmatizer = WordNetLemmatizer()
    toreturn = ' '.join(wordnet_lemmatizer.lemmatize(word, pos="v") for word in txt.split())
    return toreturn

def readify_text(txt):
    txt = clean_txt(txt)
    txt = remove_stopwords(txt)
    # txt = our_stemmer(txt)
    txt = our_lemmatizer(txt)
    return txt

def our_clf(tfidf_train, label_train, tfidf_valid, label_valid, model_filename, vectorizer, vectorizer_filename):
    clf_model = linear_model.Lasso(alpha=0.1)
    clf_model.fit(tfidf_train, label_train)
    # label_prediction = reg.predict(tfidf_valid)
    # predict the labels for the text validation data
    # label_prediction = clf_model.score(tfidf_valid)

    # check model accuracy
    #try:
    #    score = reg.score(label_valid, label_prediction)
    #    print(score)
    #except:
    #    print("Accuracy:", metrics.accuracy_score(label_valid, label_prediction))
    #print(score)

    # save trained model
    joblib.dump(clf_model, model_filename)

    # save vectorizer
    joblib.dump(vectorizer, vectorizer_filename)

    print('finishing clf')
    return clf_model


'''
    Given readied data for training and validation, do:
        Perform training, evaluate model on validation set, print confusion matrix,
        and save trained model in file 'joblib_filename'

'''
def our_svm(tfidf_train, label_train, tfidf_valid, label_valid, model_filename, vectorizer, vectorizer_filename):
    # create a svm classifier
    svm_model = svm.SVC(kernel='linear')

    #train
    svm_model.fit(tfidf_train, label_train)

    #predict the labels for the text validation data
    label_prediction = svm_model.predict(tfidf_valid)

    #check model accuracy
    print("Accuracy:", metrics.accuracy_score(label_valid, label_prediction))
    print(confusion_matrix(label_valid, label_prediction))
#bla
    #save trained model
    joblib.dump(svm_model, model_filename)

    #save vectorizer
    joblib.dump(vectorizer, vectorizer_filename)

    print('finishing svm')
    return svm_model

'''
    Load and return trained model from file 'filename'
'''
def load_trained_svm_model(filename):
    return joblib.load(filename)

'''
    Clean text:
    1. Remove links
    2. Readify text (clean, remove stopwords, stem)
    3. Replace covid-19/coronavirus with 'Corona', as hebrew speakers write 
'''
def clean_text(txt):
    txt = str(txt)
    ret = ' '.join(item for item in txt.split() if ((not (item.startswith('https://'))) and (not '.com' in item)))
    ret = readify_text(ret)
    ret = ret.replace('covid19', 'corona')
    ret = ret.replace('coronavirus', 'corona')
    ret = ret.replace('coronavir', 'corona')
    return ret

'''
    Given a dataframe with Text column, remove all links from this column
'''
def clean_dataset(df):
    df['clean_text'] = df.apply(lambda row: clean_text(row['Text']), axis=1)

def csv_cleaner(file):
    df = pd.read_csv(file)
    df = df.drop_duplicates()
    clean_dataset(df)
    new_name = file.rsplit(".", 1)[0] + 'Clean.csv'
    df.to_csv(new_name, encoding='utf-8', index=False, mode='w+')

def clean_datasets():
    csv_cleaner('trueNews.csv')
    csv_cleaner('fakeNews.csv')
    csv_cleaner('Constraint_Train.csv')
    csv_cleaner('Constraint_Val.csv')
    csv_cleaner('english_test_with_labels.csv')
    csv_cleaner('corona_fake.csv')

def training_process():
    #********************** COMBINED DAASETS **********************************
    df_true = pd.read_csv('trueNewsClean.csv')
    df_false = pd.read_csv('fakeNewsClean.csv')
    df_Constraint = pd.read_csv('Constraint_TrainClean.csv')
    df_ConstraintVal = pd.read_csv('Constraint_ValClean.csv')
    df_en_test = pd.read_csv('english_test_with_labelsClean.csv')
    df_corona_fake = pd.read_csv('corona_fakeClean.csv')
    df_true.dropna(inplace=True)
    df_false.dropna(inplace=True)
    df_Constraint.dropna(inplace=True)
    df_ConstraintVal.dropna(inplace=True)
    df_en_test.dropna(inplace=True)
    df_corona_fake.dropna(inplace=True)
    frames = [df_true, df_false, df_Constraint, df_ConstraintVal, df_en_test, df_corona_fake]
    df = pd.concat(frames, join='inner')
    df.drop_duplicates()
    labels = df['Binary Label']
    #***************************************************************************
    text_train, text_valid, label_train, label_valid = train_test_split(df['clean_text'], labels, test_size=0.2,
                                                                        random_state=109)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', ngram_range=(1, 2), norm=None)
    tfidf_train = tfidf_vectorizer.fit_transform(text_train)
    tfidf_valid = tfidf_vectorizer.transform(text_valid)
    svm_classifier = our_svm(tfidf_train, label_train, tfidf_valid, label_valid, 'combined_trained_model.pkl', tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    #clf_classifier = our_clf(tfidf_train, label_train, tfidf_valid, label_valid, 'CLF_combined_trained_model.pkl', tfidf_vectorizer, 'CLF_tfidf_vectorizer.pkl')
    our_svm_tests(tfidf_vectorizer, svm_classifier)

if __name__ == '__main__':
    #clean_datasets()  # uncomment when datasets change, or when cleaning process changes
    # training_process()  #uncomment when we want to redo training
    # *************** MANUAL CHECKS **********************
    svm_model = 'combined_trained_model.pkl'
    vectorizer = 'tfidf_vectorizer.pkl'
    grade_single_post('"קבלו רמז: אני לא התחסנתי בכלל ולא נדבקתי ולא הייתי חולה. ביי "', svm_model, vectorizer)
    grade_single_post("חבל כל חיסון מגביר את הסיכוי להידבק עוד הפעם", svm_model, vectorizer)
    grade_single_post("שאלה בורה, החיסון ידוע לכל אחד שלא מונע הדבקה", svm_model, vectorizer)
    grade_single_post("השגרירות ההודית בטוקיו אמרה שיותר מחבר צוות הודי אחד על נסיכת היהלום נמצא חיובי לקורונה", svm_model, vectorizer)
    grade_single_post("Just in: Novel coronavirus named 'Corona': UN health agency. (AFP)", svm_model, vectorizer)
    grade_single_post("WHO officially names coronavirus as Corona. CoronaOutbreak", svm_model, vectorizer)
    grade_single_post("The Indian Embassy in Tokyo has said that one more Indian crew member on Diamond Princess has tested positive for Corona.", svm_model, vectorizer)
    grade_single_post("קורונה גורמת לתחלואה קשה ותמותה גם בקרב צעירים שאינם מחוסנים 📈 עד גיל 50 - מי שלא התחסנו כלל, מהווים 69% מהחולים קשה ו-96% מהנפטרים (בין התאריכים 1.6 ל-4.9). אלו העובדות. חיסון ראשון, שני או שלישי - פשוט צאו להתחסן!", svm_model, vectorizer)
    grade_single_post("קורונה זו לא רק 'מחלה של מבוגרים'! גם צעירים שלא מתחסנים חושפים עצמם למחלה קשה. 85% ממאושפזי הקורונה אשר נמצאים כעת במצב קריטי ומחוברים למכשיר אקמו - אינם מחוסנים. גילם הממוצע - 47. קורונה עלולה להיות מחלה קשה למבוגרים ולצעירים אבל בעיקר ללא מחוסנים. צאו להתחסן.", svm_model, vectorizer)
    grade_single_post("עשרות, מאות, וכנראה אלפי אנשים שהוזרקו, מתים סובלים מדלקות בלב מנכויות קשות, מדום לב חולים במחלה שהתחסנו ממנה הכל על פי עדויות ממקור ראשון שנאספות בקושי ומראות רק את קצה הקרחון בעוד שהתמונה האמיתית נשארת במחשכים, מצונזרת ומוסתרת באלימות פראית ", svm_model, vectorizer)


"""
optional classifiers to add:
1. random forest classifier (from sklearn)
2. Multinomial Naive Bayes Algorithm (need to read more about it, we're not sure how it fits in). (from: https://www.analyticsvidhya.com/blog/2021/06/build-your-own-fake-news-classifier-with-nlp/)
3. we have 5 different classifier examples in the article:
https://www.researchgate.net/profile/Marten-Risius/publication/326405790_Automatic_Detection_of_Fake_News_on_Social_Media_Platforms/links/5b7df935a6fdcc5f8b5de39c/Automatic-Detection-of-Fake-News-on-Social-Media-Platforms.pdf
(page 11)
we want to create different functions for each classifier, and somehow work with the different results to get a more accurate analysis
"""

'''
pages talking about covid and vaccines:
FAKE:
1. mor sagmon: https://www.facebook.com/mor.sagmon
2. https://www.facebook.com/groups/VaccineChoiceIL/
3. https://www.facebook.com/groups/173406684888542/

REAL:
1. https://www.facebook.com/groups/440665513171433/about
'''