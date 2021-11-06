from Analyzer.analyzers import PotentialFakeNewsAnalysis, UTVAnalysis, machineLeaningAnalyzer
from modules.Scan_result import ScanResult
from modules.AnalysisResult import AnalysisResult
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
    sentimentAnalyzer_result = PotentialFakeNewsAnalysis.analyze_sentiments(posts)
    machine_learning_result = machineLeaningAnalyzer.grading_posts(posts)
    return ScanResult(name, sentimentAnalyzer_result, machine_learning_result, utv_result)


def analyze_post(post_obj):
    post = post_obj.content
    name = post_obj.writer
    comments_arr = [comm['Text'] for comm in post_obj.comments]
    comments_arr.insert(0, post)
    print(comments_arr)
    utv_result =  UTVAnalysis.analyze_user(post_obj.account)
    if len(comments_arr) == 0:
       return ScanResult(name, -1, -1, utv_result)
    sentimentAnalyzer_result = PotentialFakeNewsAnalysis.analyze_sentiments(comments_arr)
    machine_learning_result = machineLeaningAnalyzer.grading_posts(comments_arr)
    return ScanResult(name, sentimentAnalyzer_result, machine_learning_result, utv_result)
