from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModel, pipeline
tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_sentiment_analysis") #same as 'avichr/heBERT' tokenizer
model = AutoModel.from_pretrained("avichr/heBERT_sentiment_analysis")

# how to use?
sentiment_analysis = pipeline( "sentiment-analysis",
    model="avichr/heBERT_sentiment_analysis",
    tokenizer="avichr/heBERT_sentiment_analysis",
    return_all_scores = True
)
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
    print(englishText)
    sentimentDict = sid.polarity_scores(englishText)    # get sentiments of text
    print(sentimentDict)
    print(sentiment_analysis(post))
    if sentimentDict['pos'] < 0.1  or sentimentDict['neg'] < 0.1:
        return abs(sentimentDict['compound'])
    sum_pos_neg = sentimentDict['pos'] + sentimentDict['neg']
    return sum_pos_neg
