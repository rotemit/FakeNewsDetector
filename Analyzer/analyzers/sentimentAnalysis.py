import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
sid = SentimentIntensityAnalyzer()

def analyze_sentiments(posts):
    potentialFakePostsNum = 0
    postsNum = len(posts) # get total posts num  

    for post in posts:
        if post is not None:
            if check_fake_potential(post):
                potentialFakePostsNum += 1

    # calculate rate
    potentialFakeRate = potentialFakePostsNum / postsNum
    potentialFakeRate = 1- potentialFakeRate
    return potentialFakeRate

# check if a post might be fake by analyzing it's polarity
# idea:
# auto check >= high threshold --- return true
# auto check >= mid threshold && manual check >= super high threshold -- return true
# algo: 
# get auto calculated sentiments
# if sentiments pass high treshold - return true
# if sentiments pass mid threshold - check also manually
# if manual chack pass the super high threshold - return true
def check_fake_potential(post):  
    fake_threshold_super_high = 0.8
    fake_threshold_high = 0.7
    fake_threshold_mid = 0.5
    englishText = GoogleTranslator(source='he', target='en').translate(post)
    # auto analysis by nltk
    if englishText is None:
        return False
    sentimentDict = sid.polarity_scores(englishText)    # get sentiments of text
    # check if sentiments indicates high fake potential
    if sentimentDict['neg'] >= fake_threshold_high or sentimentDict['pos'] >= fake_threshold_high:
        return True
    
    # check if sentiments indicates mid fake potential
    elif sentimentDict['neg'] >= fake_threshold_mid or sentimentDict['pos'] >= fake_threshold_mid or abs(sentimentDict['compound']) >= fake_threshold_super_high:
        # manual analysis
        manualSentimentCalc = analyze_manualy_sentiments_in_post(englishText) # get sentiments balance by counting words
        if abs(manualSentimentCalc) >= fake_threshold_super_high:
            return True

    return False   

# analyze sentiments manually - by counting words
def analyze_manualy_sentiments_in_post(englishText):
    pos_word_list = []
    neg_word_list = []
    neu_word_list = []
    wordList = re.sub("[^\w]", " ", englishText).split()
    
    for word in wordList:
        if (sid.polarity_scores(word)['compound']) >= 0.1:
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.1:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)

    countPos = len(pos_word_list)
    countNeg = len(neg_word_list)
    countNeu = len(neu_word_list)
    countTotal = countNeg + countPos + countNeu
    sentimentCalc = 0

    if(countPos == countNeg):
        sentimentCalc = 0
    elif(countPos > countNeg):
        sentimentCalc = countPos / (countPos + countNeg)
    else:
        sentimentCalc = countNeg / (countPos + countNeg) * (-1)
    return sentimentCalc
