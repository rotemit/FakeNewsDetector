from Analyzer.analyzers.machineLeaningAnalyzer import grading_posts
from Analyzer.analyzers import PotentialFakeNewsAnalysis, UTVAnalysis
from modules.Scan_result import ScanResult
from modules.AnalysisResult import AnalysisResult


############ analysis manager ############

# gets account object, performs all analysis, and returns results as ScanResult object
def analyze_account(account):
    posts = account.posts
    utv_result = UTVAnalysis.analyze_user(account)
    if len(posts)==0:
        # result for none posts profile
        return create_not_enough_posts_result(account, "This user doesn't have any posts, hence does not have result.", utv_result)

    elif len(posts)<=5:
        # result for not enough posts profile
        return create_not_enough_posts_result(account, "This user doesn't have enough posts to derive conclusions from.", utv_result)

    else:
        # perform all analyses
        sentimentAnalyzer_result = PotentialFakeNewsAnalysis.analyze_user(account)
        machine_learning_result :AnalysisResult= grading_posts(posts)
        machine_learning_result.text = "The account " + machine_learning_result.text

    return ScanResult(account.name, sentimentAnalyzer_result, machine_learning_result, utv_result)

def analyze_page(page):
    posts = page.posts
    utv_result = AnalysisResult("N\A", "No Trust Value result to Facebook Page", 0)

    if len(posts)==0:
        # result for none posts page
        return create_not_enough_posts_result(account, "This page doesn't have any posts, hence does not have result.", utv_result)

    elif len(posts)<=5:
        # result for not enough posts page
        return create_not_enough_posts_result(account, "This page doesn't have enough posts to derive conclusions from.", utv_result)

    else:
        # perform all analyses
        sentimentAnalyzer_result = PotentialFakeNewsAnalysis.analyze_user(page)
        machine_learning_result: AnalysisResult = grading_posts(posts)
        machine_learning_result.text = "The page " + machine_learning_result.text

    return ScanResult(page.name, sentimentAnalyzer_result, machine_learning_result, utv_result)


def analyze_group(group):
    posts = group.posts
    utv_result = AnalysisResult("N\A", "No Trust Value result to Facebook Group", 0)

    if len(posts) == 0:
        # result for none posts group
        return create_not_enough_posts_result(account, "This group doesn't have any posts, hence does not have result.",utv_result)

    elif len(posts) <= 5:
        # result for not enough posts group
        return create_not_enough_posts_result(account,"This group doesn't have enough posts to derive conclusions from.", utv_result)

    else:
        # perform all analyses
        sentimentAnalyzer_result = PotentialFakeNewsAnalysis.analyze_user(group)
        machine_learning_result: AnalysisResult = grading_posts(posts)
        machine_learning_result.text = "The group " + machine_learning_result.text

    return ScanResult(group.name, sentimentAnalyzer_result, machine_learning_result, utv_result)


# create result for profile with not enough posts (0 posts or not enough)
def create_not_enough_posts_result(to_analyze, text_result, utv_result):
    text_analyzers_result = AnalysisResult("N\A", text_result, 0)
    result = ScanResult(to_analyze.name, text_analyzers_result, text_analyzers_result, utv_result)
    return result
