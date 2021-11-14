from Analyzer.analyzers import sentimentAnalysis, UTVAnalysis, machineLeaningAnalyzer
from modules.Scan_result import ScanResult
from modules.Post import Post

############ analysis manager ############

def analyze_facebook(obj):
    if obj is None:
        return None
    if isinstance(obj, Post):
        return analyze_post(obj)
    posts = obj.posts
    name = obj.name
    utv_result = UTVAnalysis.analyze_user(obj)
    if len(posts) == 0:
       return ScanResult(name, -1, -1, utv_result)
    sentimentAnalyzer_result = sentimentAnalysis.analyze_sentiments(posts)
    machine_learning_result = machineLeaningAnalyzer.grading_posts(posts)
    return ScanResult(name, sentimentAnalyzer_result, machine_learning_result, utv_result)


def analyze_post(post_obj):
    post = post_obj.content
    name = post_obj.writer
    if post_obj.account is None and post is not None:
        sentimentAnalyzer_result = sentimentAnalysis.analyze_sentiments([post])
        machine_learning_result = machineLeaningAnalyzer.grading_posts([post])
        return ScanResult(name, sentimentAnalyzer_result, machine_learning_result, -1)

    posts = post_obj.account.posts
    if post is not None:
        posts.insert(0, post)
    utv_result =  UTVAnalysis.analyze_user(post_obj.account)
    if len(posts) == 0:
        return ScanResult(name, -1, -1, utv_result)
    sentimentAnalyzer_result = sentimentAnalysis.analyze_sentiments(posts)
    machine_learning_result = machineLeaningAnalyzer.grading_posts(posts)
    return ScanResult(name, sentimentAnalyzer_result, machine_learning_result, utv_result)


def analyze_string(txt):
    sentimentAnalyzer_result = sentimentAnalysis.analyze_sentiments([txt])
    machine_learning_result = machineLeaningAnalyzer.grading_posts([txt])
    return ScanResult("Text", sentimentAnalyzer_result, machine_learning_result, -1)