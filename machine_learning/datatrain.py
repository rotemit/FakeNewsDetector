import csv

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
from sklearn.feature_extraction.text import HashingVectorizer

from deep_translator import GoogleTranslator #pip installed


#manual data for testing
data = [
    # ['1', 'ילדים חכמים אוהבים שוקולד'],
    # ['2', 'ילדים חכמים אוהבים ילדים חכמים'],
    # ['1', 'ילדות חכמות אוהבות שוקולד'],
    # ['1', 'ילדים חכמים אוהבים לחקור שוקולד'],
    # ['2', 'ילדות חכמות אוהבות רצים מהר'],
    # ['1', 'ילדות אוהבות רצים']
]


'''
    posts we took from facebook for testing
'''
covid_posts = [
    ['"לא ידענו שזה השלום האחרון": יותר מ-550 חולי קורונה נפטרו החודש בישראל. בני המשפחה, החברים וגם בעלי החיים לא מפסיקים להתגעגע'], #kan news
    ['מבצע החיסונים מוצלח מאוד, הן בקצבו והן בתוצאותיו. לאור זאת, אנחנו מרחיבים את האפשרות לקבל חיסון שלישי לכלל האוכלוסייה, ובתנאי שחלפו 5 חודשים מאז המנה השנייה. צאו להתחסן, בין אם זה חיסון ראשון, שני או שלישי - זה חשוב לכל אחד מכם וזה חשוב לכולנו כאוכלוסייה!'], #health ministry
    ['תראו איך מרעילים את הילדים באמצעות הבדיקה ששלחו לנו מבתי הספר והגנים!!!!🙄 לא לעשות!!!!'], #נפגעי חיסון הקורונה
    ['פייזר. ארגון פשע חוקי. שוחד, הסתרת נזקים, ייצור והפצה של סמים פסיכיאטריים ושאר רעלים במסווה של ״רפואה״ וקנסות של מאות מיליוני דולרים (רק על מה שהם נתפסו).'], #mor sagmon page
    ['אז לגבי המרכיבים שבזריקות, מסתבר שישנם רכיבים שלא דווחו'],  #vaccine choice il
    ['פנינים שנאמרו בזום המורים. ושימו לב, 25% מבעלי התו הירוק קיבלו פלסיבו'],  #vaccine choice il
    ['קונספירציה חדשה: הבדיקות המהירות שחילקו בבתי הספר הם לא ל covid 19. עבדו עלינו, זה לסארס 2!!!!! אנשים הזויים. יכול להבין שאנשים לא יודעים סתם ככה, אבל מה קשה לעשות גוגל?!'],  #מה כבר יכול לקרות
    ['קורונה מסוכנת וגורמת למוות'],  #rotem
    ['Covid-19 גורמת למוות'],  #rotem
    ['ארגון הבריאות העולמי מכנה רשמית את הקורונה בתור קוביד-19'],  #
    ['תמונה מראה אנשים שנדבקו בקורונה שוכבים על המדרכה בסין'],  #
    ['כשבועיים מאז החל מבצע החיסונים בחיסון השלישי, חצתה היום מדינת ישראל את רף מיליון המתחסנים בחיסון השלישי. מדובר ביותר ממחצית האוכלוסייה ברת החיסון (כ-1.9 מיליון בני 50+, שחלפו חמישה חודשים מאז קיבלו את מנת החיסון השנייה).'],  #
    ['ההבדלים בין החיסון של מודרנה לבין זה של פייזר קטנים מאד מכל הבחינות. אותה טכנולוגיה ויעילות ובטיחות דומים. לגבי דלקת בשריר הלב - רק אצל הרופא והקרדיולוג.'], #מדברימדע בתגובות
  #  [' דלקת שריר הלב כתופעת לוואי של החיסון היא נדירה מאוד. לעומת זאת היא נפוצה למדי אצל חולי קורונה. גם אני אגיד שכדאי לשאול את הרופא, אבל אני מאוד אופתע אם הוא לא ימליץ בחום להתחסן. למיטב ידיעתי, המצבים בהם לא מומלץ להתחסן נדירים וקשורים לעניינים של מערכת החיסון ומחלות אוטו-אימוניות, לא ללב.'],  ##מדברימדע בתגובות
    # [''],  #
    # [''],  #
    # [''],  #
    # [''],  #
]

'''
    manual tests for svm
'''
def our_svm_tests(tfidf_vectorizer, svm_classifier):
    for post in covid_posts:
        en_post = GoogleTranslator(source='he', target='en').translate(post[0])
        grade = grade_post(en_post, tfidf_vectorizer, svm_classifier)
        print('Post: '+en_post+'\nGrade: '+str(grade)+'\n')


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

    return svm_clf


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

# def my_csv_modifier():
#     with open('Constraint_Train.csv', 'a') as f:


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
    # my_csv_writer()
    # df = pd.read_csv('my_csv_file_lemmatized.csv')


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
    # csv_cleaner('trueNews.csv')     #create a new and clean csv file. uncomment only when file changes
    # csv_cleaner('fakeNews.csv')     #create a new and clean csv file
    # csv_cleaner('Constraint_Train.csv')
    df_true = pd.read_csv('trueNewsClean.csv')
    df_false = pd.read_csv('fakeNewsClean.csv')
    df_Constraint = pd.read_csv('Constraint_TrainClean.csv')
    # df_false['our_labels'] = df_false.apply(lambda col: col['Poynter_Label'].upper(), axis=1)
    df_true.dropna(inplace=True)
    df_false.dropna(inplace=True)
    df_Constraint.dropna(inplace=True)
    frames = [df_true, df_false, df_Constraint]
    df = pd.concat(frames, join='inner')

    labels = df['Binary Label']

    #***************************************************************************

    #********************************Vaccine Tweets Datasets*******************************
    # csv_cleaner('vaccination_tweets.csv')
    # df = pd.read_csv('Constraint_Train.csv')
    # labels = df['label']
    #***************************************************************************************


    text_train, text_valid, label_train, label_valid = train_test_split(df['clean_text'], labels, test_size=0.2,
                                                                        random_state=109)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', ngram_range=(1, 1), norm=None)
    tfidf_train = tfidf_vectorizer.fit_transform(text_train)
    tfidf_valid = tfidf_vectorizer.transform(text_valid)
    svm_classifier = our_svm(tfidf_train, label_train, tfidf_valid, label_valid)

    our_svm_tests(tfidf_vectorizer, svm_classifier)



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

'''
pages talking about covid and vaccines:
FAKE:
1. mor sagmon: https://www.facebook.com/mor.sagmon
2. https://www.facebook.com/groups/VaccineChoiceIL/
3. https://www.facebook.com/groups/173406684888542/

REAL:
1. https://www.facebook.com/groups/440665513171433/about
'''