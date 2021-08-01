import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from googletrans import Translator

if __name__ == '__main__':
    # Read the data
    df = pd.read_csv('mashrokit.csv')
    # Get shape and head
    df.shape
    df.head()
    # DataFlair - Get the labels
    labels = df.label
    labels.head()
    # DataFlair - Split the dataset WE DONT WANT THIS BECAUSE WERE ONLY LOOKING FOR TRAINING DATA
    # x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    # DataFlair - Initialize a TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7)

    # DataFlair - Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    # DataFlair - Initialize a PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    # testing with posts from the shadow. the two last ones are about the olympics, not correlating with our training set but with interesting results
    post = remove_stopwords("זה מה שמצאו מתפללים בשני בתי כנסת אתמול  בבני ברק. לבית הכנסת נזרקו גם קונדומים תמונות פורנגרפיות ותמונות של שירה בנקי זל שנרצחה במצעד הגאווה. אין מה להגיד יש הרגשה של ריפוי באויר.")
    print("psot: " + post)
    grade = grade_post(post, tfidf_vectorizer, pac)
    print(grade)
    post = remove_stopwords("פיטר פלצ'יק לאחר הזכיה אתמול במדלית ארד  באולימפיאדה היום נלחמתי לא רק בשביל עצמי ולא רק בשביל המטרות שלי והחלומות שלי, אני נלחמתי  בשביל הקבוצה, בשביל הלב שלנו, המדינה שלנו, בשביל הדגל הזה, ואני לא הולך להוריד אותו בשעות הקרובות והוא יהיה השמיכה שלי היום בלילה.")
    grade = grade_post(post, tfidf_vectorizer, pac)
    print(grade)
    post = remove_stopwords("התקווה הושמעה בטוקיו!!! תנו המון כבוד לארטיום דולגופיאט שזכה במדליית הזהב בתרגיל הקרקע באולימפיאדת טוקיו 2020. זהו ההישג הגדול ביותר לספורט הישראלי בכל הזמנים: מדליית זהב אולימפית ראשונה לישראל מאז אתונה 2004, והראשונה אי פעם באחד מענפי החשובים ביותר של המשחקים.")
    print(post)
    grade = grade_post(post, tfidf_vectorizer, pac)
    print(grade)
    #testing with a political post by miri regev
    post = remove_stopwords("דמיינו, שאתם הייתה השכנים של גדעון סער והיה לכם סכסוך נגיד על חנייה. יום למחרת גדעון סער כשר המשפטים היה מעביר חוק שפוגע בדיוק בכם באותו סכסוך על חנייה. גדעון סער מונע ממסע נקמה אישי נגד בנימין נתניהו, יש כאן ניגוד עניינים ברור והוא לא יכול להתעסק בשום הצעת חוק הקשורה לנתניהו. ")
    grade = grade_post(post, tfidf_vectorizer, pac)
    print("Miri Regev's post grade: " + str(grade))
    #merav michaeli post
    post = remove_stopwords("הבריאות שלנו היא מעל הכל. סיכמתי עם ראש הממשלה נפתלי בנט - Naftali Bennett ועם משרד האוצר על גיוס של 400 פקחים אשר יאכפו את עטיית המסיכות בתחבורה הציבורית כדי למנוע הדבקה. המלחמה בקורונה היא לטובת כולנו - אל תזלזלו והקפידו על עטיית מסיכה. ")
    grade = grade_post(post, tfidf_vectorizer, pac)
    print(grade)


    # DataFlair - Predict on the test set and calculate accuracy
    #   y_pred = pac.predict(tfidf_test)
    # score = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {round(score * 100, 2)}%')
    # # DataFlair - Build confusion matrix
    # print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5]))

"""
optional classifiers to add:
1. random forest classifier (from sklearn)
2. Multinomial Naive Bayes Algorithm (need to read more about it, we're not sure how it fits in). (from: https://www.analyticsvidhya.com/blog/2021/06/build-your-own-fake-news-classifier-with-nlp/)
3. we have 5 different classifier examples in the article:
https://www.researchgate.net/profile/Marten-Risius/publication/326405790_Automatic_Detection_of_Fake_News_on_Social_Media_Platforms/links/5b7df935a6fdcc5f8b5de39c/Automatic-Detection-of-Fake-News-on-Social-Media-Platforms.pdf
(page 11)
we want to create different functions for each classifier, and somehow work with the different results to get a more accurate analysis
"""