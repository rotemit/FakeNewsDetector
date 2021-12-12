from Analyzer.analyzers import sentimentAnalyzer, trustValueAnalyzer, machineLeaningAnalyzer
from modules.Scan_result import ScanResult
from modules.Post import Post
import pandas as pd
############ analysis manager ############

def analyze_facebook(obj, file=None):
    if obj is None or isinstance(obj, str):
        return None

    if isinstance(obj, Post):
        return analyze_post(obj)

    posts = obj.posts
    name = obj.name
    df = None
    utv_result = trustValueAnalyzer.analyze_facebook(obj)
    if file is not None:
        df = pd.DataFrame(columns=['Trust Value', 'Sentiment Analysis', 'Machine Learning', 'Post'], index=range(len(posts)))
        for i in range(len(posts)):
            df.iloc[i, 0] = utv_result

    if len(posts) == 0:
       return ScanResult(name, -1, -1, utv_result)

    sentimentAnalyzer_result = sentimentAnalyzer.analyze_sentiments(posts, df)
    machine_learning_result = machineLeaningAnalyzer.grading_posts(posts, df)
    if file is not None:
        df.to_csv(file, index=False)
    return ScanResult(name, sentimentAnalyzer_result, machine_learning_result, utv_result)


def analyze_post(post_obj):
    post = post_obj.content
    name = post_obj.writer

    utv_result = -1
    sentimentAnalyzer_result = -1
    machine_learning_result = -1

    if post_obj.account is not None:
        utv_result = trustValueAnalyzer.analyze_facebook(post_obj.account)

    if post is not None and post != "":
        sentimentAnalyzer_result = sentimentAnalyzer.analyze_sentiments([post])
        machine_learning_result = machineLeaningAnalyzer.grading_posts([post])

    return ScanResult(name, sentimentAnalyzer_result, machine_learning_result, utv_result)


def analyze_string(txt):
    if txt is None or txt == "":
        return ScanResult("Text", -1, -1, -1)

    sentimentAnalyzer_result = sentimentAnalyzer.analyze_sentiments([txt])
    machine_learning_result = machineLeaningAnalyzer.grading_posts([txt])
    return ScanResult("Text", sentimentAnalyzer_result, machine_learning_result, -1)
