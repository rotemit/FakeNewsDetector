from machine_learning.datatrain import grade_single_post
from modules.AnalysisResult import AnalysisResult
from Analyzer.analyzers.CovidWords import covid_list

COVID_PERCENT_ABOVE = 0 #may be changed after more analysis
def grading_posts(posts):
    amount = 0
    counter = 0
    model = "machine_learning/combined_trained_model.pkl"
    vectorizer = "machine_learning/tfidf_vectorizer.pkl"
    for post in posts:
        if(check_covid_relateness(post) > COVID_PERCENT_ABOVE):
            counter += grade_single_post(post, model, vectorizer)[0]
            amount += 1
    if amount == 0:
       return AnalysisResult("N\A", "is not talking about Covid 19", 1)

    grade = counter / amount
    percent = int((grade * 100) // 1)
    percentResult = str(percent) + "%"
    return AnalysisResult(percentResult, convert_potential_fake_rate_to_text(grade), grade)

def convert_potential_fake_rate_to_text(potentialFakeRate):
    for rate in MLAnalysisTextResult.keys():
        if potentialFakeRate <= rate:
            return MLAnalysisTextResult[rate]
    return ""

def check_covid_relateness(post):
    counter = 0
    length = len(post.split(' '))
    for word in covid_list:
        if word in post:
            counter += 1
    div = counter/length
    percent = int((div * 100) // 1)
    print("\n" +post)
    print(div)
    print(percent)
    return percent

MLAnalysisTextResult = {
    0.0: "is DANGEROUS! All posts are potential fake news!",
    0.2: "is problematic, the vast majority of posts are potential fake news!",
    0.4: "is problematic, most posts are potential fake news.",
    0.6: "often post potential fake news, pay attention!",
    0.8: "rarely post potential fake news.",
    0.9: "is ok.",
    1.0: "is clean of potential fake news!"
}