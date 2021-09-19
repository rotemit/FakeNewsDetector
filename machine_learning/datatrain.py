import csv
#testtt
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import stop_words
from yap_server import get_lemma
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.feature_extraction.text import HashingVectorizer

from deep_translator import GoogleTranslator #pip installed
from heb_data_collector import get_group_posts


def grade_single_post(post):
    return -1
    # svm_model = joblib.load('combined_trained_model.joblib')
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', ngram_range=(1, 1), norm=None)
    # tfidf_train = tfidf_vectorizer.fit_transform(text_train)
    # tfidf_valid = tfidf_vectorizer.transform(text_valid)
    # tfidf_test = tfidf_vectorizer.transform([post])
    # y_pred = svm_model.predict(tfidf_test)
    # print(y_pred)

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

'''
    Given readied data for training and validation, either:
        1. Perform training, evaluate model on validation set, print confusion matrix,
           and save trained model in file 'joblib_filename'
        OR
        2. Load and return trained model from file 'joblib_filename'
'''
def our_svm(tfidf_train, label_train, tfidf_valid, label_valid, joblib_filename):
    #*************TRAIN AND SAVE*************
    # # create a svm classifier
    # svm_model = svm.SVC(kernel='linear')
    #
    # #train
    # svm_model.fit(tfidf_train, label_train)
    #
    # #predict the labels for the text validation data
    # label_prediction = svm_model.predict(tfidf_valid)
    #
    # #check model accuracy
    # print("Accuracy:", metrics.accuracy_score(label_valid, label_prediction))
    # print(confusion_matrix(label_valid, label_prediction))
    #
    # #save trained model
    # joblib.dump(svm_model, joblib_filename)
    #**************FINISH TRAIN AND SAVE*********

    #when using a pre-trained model, comment previous code and uncomment next line
    svm_model = joblib.load(joblib_filename)
    print('finishing svm')
    return svm_model


'''
    Clean text:
    1. Remove links
    2. Replace covid-19/coronavirus with 'Corona', as hebrew speakers write 
'''
def clean_text(txt):
    ret = ' '.join(item for item in txt.split() if ((not (item.startswith('https://'))) and (not '.com' in item)))
    ret = ret.replace('COVID-19', 'Corona')
    ret = ret.replace('Covid-19', 'Corona')
    ret = ret.replace('Coronavirus', 'Corona')
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

if __name__ == '__main__':
    # *************** TRYING COVID with the two files ***************
    # # csv_cleaner('trueNews.csv')     #create a new and clean csv file. uncomment only when file changes
    # # csv_cleaner('fakeNews.csv')     #create a new and clean csv file
    # print('1')
    # df_true = pd.read_csv('trueNewsClean.csv')
    # print('2')
    # df_false = pd.read_csv('fakeNewsClean.csv')
    # print('3')
    # # df_false['our_labels'] = df_false.apply(lambda col: col['Poynter_Label'].upper(), axis=1)
    # df_true.dropna(inplace=True)
    # print('4')
    # df_false.dropna(inplace=True)
    # print('5')
    # frames = [df_true, df_false]
    # df = pd.concat(frames, join='inner')
    #
    # labels = df['Binary Label']
    #******************************************************

    #*********************DATASET: Constraint_Train *********************
    # df = pd.read_csv('Constraint_TrainClean.csv')
    # labels = df['label']
    #*********************************************************************

    #********************** COMBINED DAASETS **********************************
    # # csv_cleaner('trueNews.csv')     #create a new and clean csv file. uncomment only when file changes
    # # csv_cleaner('fakeNews.csv')     #create a new and clean csv file
    # # csv_cleaner('Constraint_Train.csv')
    # # csv_cleaner('Constraint_Val.csv')
    # # csv_cleaner('english_test_with_labels.csv')
    # df_true = pd.read_csv('trueNewsClean.csv')
    # df_false = pd.read_csv('fakeNewsClean.csv')
    # df_Constraint = pd.read_csv('Constraint_TrainClean.csv')
    # df_ConstraintVal = pd.read_csv('Constraint_ValClean.csv')
    # df_en_test = pd.read_csv('english_test_with_labelsClean.csv')
    # # df_false['our_labels'] = df_false.apply(lambda col: col['Poynter_Label'].upper(), axis=1)
    # df_true.dropna(inplace=True)
    # df_false.dropna(inplace=True)
    # df_Constraint.dropna(inplace=True)
    # df_ConstraintVal.dropna(inplace=True)
    # df_en_test.dropna(inplace=True)
    # frames = [df_true, df_false, df_Constraint, df_ConstraintVal, df_en_test]
    # df = pd.concat(frames, join='inner')
    # df.drop_duplicates()
    #
    # labels = df['Binary Label']

    #***************************************************************************


    # text_train, text_valid, label_train, label_valid = train_test_split(df['clean_text'], labels, test_size=0.2,
    #                                                                     random_state=109)
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', ngram_range=(1, 1), norm=None)
    # tfidf_train = tfidf_vectorizer.fit_transform(text_train)
    # tfidf_valid = tfidf_vectorizer.transform(text_valid)
    # svm_classifier = our_svm(tfidf_train, label_train, tfidf_valid, label_valid, 'combined_trained_model.joblib')
    #
    # our_svm_tests(tfidf_vectorizer, svm_classifier)



    # *************** FINISH **********************
    grade_single_post('"קבלו רמז: אני לא התחסנתי בכלל ולא נדבקתי ולא הייתי חולה. ביי "')
    grade_single_post("חבל כל חיסון מגביר את הסיכוי להידבק עוד הפעם")
    grade_single_post("שאלה בורה, החיסון ידוע לכל אחד שלא מונע הדבקה")




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