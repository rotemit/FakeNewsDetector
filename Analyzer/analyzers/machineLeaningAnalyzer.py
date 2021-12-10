import pandas as pd
from machine_learning.datatrain import grade_single_post
from Analyzer.analyzers.CovidWords import covid_list


def grading_posts(posts, df=None):
    amount = 0
    counter = 0
    model = "machine_learning\BERT_model.pkl"
    tokenizer = "machine_learning\AlephBERT_tokenizer.pkl"
    for i, post in enumerate(posts):
        if(check_covid_relateness(post) > 0):
            post_grade = grade_single_post(post, model, tokenizer)
            counter += post_grade
            amount += 1
            if df is not None:
                df.iat[i, 2] = post_grade
        elif df is not None:
            df.iat[i, 2] = "N/A"
    if amount == 0:
       return -1

    grade = counter / amount
    return grade


def check_covid_relateness(post):
    if post is None or post == "":
        return 0
    lower_post = post.lower()
    counter = 0
    length = len(post.split(' '))
    for word in covid_list:
        if word in lower_post:
            counter += 1
    div = counter/length
    percent = int((div * 100) // 1)
    return percent
