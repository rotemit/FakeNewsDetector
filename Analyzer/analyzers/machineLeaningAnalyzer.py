from machine_learning.datatrain import grade_single_post
from Analyzer.analyzers.CovidWords import covid_list

def grading_posts(posts):
    amount = 0
    counter = 0
    model = "machine_learning/combined_trained_model.pkl"
    vectorizer = "machine_learning/tfidf_vectorizer.pkl"
    for post in posts:
        if(check_covid_relateness(post) > 0):
            # counter += grade_single_post(post, model, vectorizer)[0]
            counter += 1 #TODO: change when ML works
            amount += 1
    if amount == 0:
       return -1

    grade = counter / amount
    return grade


def check_covid_relateness(post):
    counter = 0
    length = len(post.split(' '))
    for word in covid_list:
        if word in post:
            counter += 1
    div = counter/length
    percent = int((div * 100) // 1)
    return percent
