from Analyzer.analyzers import PotentialFakeNewsAnalysis, UTVAnalysis, machineLeaningAnalyzer
from modules.Scan_result import ScanResult
from modules.AnalysisResult import AnalysisResult


############ analysis manager ############

# gets account object, performs all analysis, and returns results as ScanResult object
def analyze_account(account):
    posts = account.posts
    utv_result = UTVAnalysis.analyze_user(account)
    if len(posts)<=5:
        # result for not enough posts profile
        return create_not_enough_posts_result(account.name, utv_result)

    else:
        # perform all analyses
        sentimentAnalyzer_result = PotentialFakeNewsAnalysis.analyze_sentiments(posts)
        # sentimentAnalyzer_result.text = "The account " + sentimentAnalyzer_result.text
        machine_learning_result = machineLeaningAnalyzer.grading_posts(posts)
        # machine_learning_result.text = "The account " + machine_learning_result.text

    return ScanResult(account.name, sentimentAnalyzer_result, machine_learning_result, utv_result)

def analyze_page(page):
    posts = page.posts
    utv_result = 1.0

    if len(posts)<=5:
        # result for not enough posts page
        return create_not_enough_posts_result(page.name, "This page doesn't have enough posts to derive conclusions from.", utv_result)

    else:
        # perform all analyses
        sentimentAnalyzer_result = PotentialFakeNewsAnalysis.analyze_sentiments(posts)
        sentimentAnalyzer_result.text = "The page " + sentimentAnalyzer_result.text
        machine_learning_result = machineLeaningAnalyzer.grading_posts(posts)
        machine_learning_result.text = "The page " + machine_learning_result.text

    return ScanResult(page.name, sentimentAnalyzer_result, machine_learning_result, utv_result)


def analyze_group(group):
    posts = group.posts
    utv_result =  1.0

    if len(posts) <= 5:
        # result for not enough posts group
        return create_not_enough_posts_result(group.name, utv_result)

    else:
        # perform all analyses
        sentimentAnalyzer_result = PotentialFakeNewsAnalysis.analyze_sentiments(posts)
        # sentimentAnalyzer_result.text = "The group " + sentimentAnalyzer_result.text
        machine_learning_result = machineLeaningAnalyzer.grading_posts(posts)
        # machine_learning_result.text = "The group " + machine_learning_result.text

    return ScanResult(group.name, sentimentAnalyzer_result, machine_learning_result, utv_result)

def analyze_post(post_obj):
    post = post_obj.content
    utv_result =  0.0
    writer_analysis = analyze_account(post_obj.account)
    sentimentAnalyzer_result = PotentialFakeNewsAnalysis.analyze_one_post(post)
    machine_learning_result = machineLeaningAnalyzer.grading_one_post(post)
    comments_analysis = analyze_comments(post_obj.comments)
    post_analysis = ScanResult(post_obj.writer, sentimentAnalyzer_result, machine_learning_result, utv_result)
    return [("Writer", writer_analysis), ("Post", post_analysis), ("Comments", comments_analysis)]

def analyze_comments(comments):
    comments_arr = [comm['Text'] for comm in comments]
    utv_result =  0.0
    if len(comments_arr) <= 5:
        # result for not enough posts group
        return create_not_enough_posts_result("Comments", utv_result)

    else:
        # perform all analyses
        sentimentAnalyzer_result = PotentialFakeNewsAnalysis.analyze_sentiments(comments_arr)
        textResult_arr = sentimentAnalyzer_result.text.split(' ')
        if textResult_arr[0] == 'is':
            textResult_arr[0] = 'are'
        textResult = ' '.join(textResult_arr)
        sentimentAnalyzer_result.text = "The comments " + textResult

        machine_learning_result = machineLeaningAnalyzer.grading_posts(comments_arr)
        textResult_arr = machine_learning_result.text.split(' ')
        if textResult_arr[0] == 'is':
            textResult_arr[0] = 'are'
        textResult = ' '.join(textResult_arr)
        machine_learning_result.text = "The comments " + textResult

    return ScanResult("Comments", sentimentAnalyzer_result, machine_learning_result, utv_result)



# create result for profile with not enough posts (0 posts or not enough)
def create_not_enough_posts_result(name, utv_result):
    result = ScanResult(name, 0.0, 1.0, utv_result)
    return result
