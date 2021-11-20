from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
sid = SentimentIntensityAnalyzer()


def analyze_sentiments(posts):
    potentialFakePostsNum = 0
    amount = 0
    for post in posts:
        if post is not None:
            intensity = check_fake_potential(post)
            potentialFakePostsNum +=  intensity
            amount += 1
            print("post: " + post + "\ngrade: " + str(intensity) + "\n")

    # calculate rate
    potentialFakeRate = potentialFakePostsNum / amount
    potentialFakeRate = 1 - potentialFakeRate #colser to 0 = more FAKE
    return potentialFakeRate

# returns the polarity both positive and negative of a post
def check_fake_potential(post):
    englishText = GoogleTranslator(source='he', target='en').translate(post)
    if englishText is None:
        return 0

    # auto analysis by nltk
    sentimentDict = sid.polarity_scores(englishText)
    #if the post is either positive or negative - return the absoulte of 'compound'
    if sentimentDict['pos'] == 0.0  or sentimentDict['neg'] == 0.0:
        return abs(sentimentDict['compound'])
    #if the post contains both positive and negative phrases - return the sum of 'pos' and 'neg'
    sum_pos_neg = sentimentDict['pos'] + sentimentDict['neg']
    return sum_pos_neg
