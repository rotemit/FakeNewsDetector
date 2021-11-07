import csv
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from machine_learning.stop_words import stop_words
from machine_learning.yap_server import get_keyWords
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import linear_model
from deep_translator import GoogleTranslator #pip installed
# from machine_learning.heb_data_collector import get_group_posts
from sklearn.linear_model import LogisticRegressionCV
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
# import nltk
# nltk.download('wordnet')
#*********imports for deep learning************
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam
import gensim

from transformers import BertModel, BertTokenizerFast
import torch.nn as nn
# from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch

#pips: pip install tensorflow --user
#       pip install gensim
#       pip install transformers    #for AlephBERT
#       pip install torch torchvision torchaudio    #for AlephBERT

# def grade_single_post(post, model, tokenizer):
#     #load trained model and fitted tokenizer
#     model = joblib.load(model)
#     tokenizer = joblib.load(tokenizer)
#     #prepare post
#     post = [post]
#
#
#     #translate post to english, regardless of source language
#     translator = GoogleTranslator()
#     translated_post = translator.translate(post)
#     readied_post = readify_text(translated_post)
#     #vectorize and predict fakeness
#     vectorized_post = tokenizer.transform([readied_post])
#     y_pred = model.predict(vectorized_post)
#     #print result
#     print("post:\n" + post + "\ntranslated post:\n" + translated_post + "\nreadied post:\n"+readied_post+"\ngrade: " + str(y_pred))
#     return y_pred

'''
    manual tests for svm
'''
def our_svm_tests(tfidf_vectorizer, svm_classifier):
    posts = pd.read_csv('heb_posts.csv')
    for post in posts:
        en_post = GoogleTranslator(source='he', target='en').translate(post)
        grade = grade_post(en_post, tfidf_vectorizer, svm_classifier)
        print('Post: '+en_post+'\nGrade: '+str(grade)+'\n')

"""
    Given a post, a _ , and a classifier,
    return a binary grade indicating the level of fake
"""
def grade_post (post, machine, classifer):
    tfidf_test = machine.transform([post])
    y_pred = classifer.predict(tfidf_test)
    return y_pred

"""
    Remove punctuation
"""
def remove_punc(txt):
    #make lowercase
    txt = txt.lower()
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in txt:
        if ele in punc:
            txt = txt.replace(ele, "")
    return txt

"""
    Remove frequent and unimportant words
"""
def remove_stopwords(txt):
    stopwords = stop_words
    tokens = txt.split(" ")
    resultwords = [word for word in tokens if word not in stopwords]
    result = ' '.join(resultwords)
    return result

def our_stemmer(txt):
    # ps = PorterStemmer()
    # toreturn = ' '.join(ps.stem(word) for word in txt.split())
    # return toreturn
    ls = LancasterStemmer()
    toreturn = ' '.join(ls.stem(word) for word in txt.split())
    return toreturn

def our_lemmatizer(txt):
    wordnet_lemmatizer = WordNetLemmatizer()
    toreturn = ' '.join(wordnet_lemmatizer.lemmatize(word, pos="v") for word in txt.split())
    return toreturn

def readify_text(txt):
    txt = remove_punc(txt)
    txt = remove_stopwords(txt)
    # txt = our_stemmer(txt)
    txt = our_lemmatizer(txt)
    return txt

def our_clf(tfidf_train, label_train, tfidf_valid, label_valid, model_filename, vectorizer, vectorizer_filename):
    clf_model = linear_model.Lasso(alpha=0.1)
    clf_model.fit(tfidf_train, label_train)
    # label_prediction = reg.predict(tfidf_valid)
    # predict the labels for the text validation data
    # label_prediction = clf_model.score(tfidf_valid)

    # check model accuracy
    #try:
    #    score = reg.score(label_valid, label_prediction)
    #    print(score)
    #except:
    #    print("Accuracy:", metrics.accuracy_score(label_valid, label_prediction))
    #print(score)

    # save trained model
    joblib.dump(clf_model, model_filename)

    # save vectorizer
    joblib.dump(vectorizer, vectorizer_filename)

    print('finishing clf')
    return clf_model


'''
    Given readied data for training and validation, do:
        Perform training, evaluate model on validation set, print confusion matrix,
        and save trained model in file 'joblib_filename'

'''
def our_svm(tfidf_train, label_train, tfidf_valid, label_valid, model_filename, vectorizer, vectorizer_filename):
    # create a svm classifier
    svm_model = svm.SVC(kernel='linear')

    #train
    svm_model.fit(tfidf_train, label_train)

    #predict the labels for the text validation data
    label_prediction = svm_model.predict(tfidf_valid)

    #check model accuracy
    print("Accuracy:", metrics.accuracy_score(label_valid, label_prediction))
    print(confusion_matrix(label_valid, label_prediction))

    #save trained model
    joblib.dump(svm_model, model_filename)

    #save vectorizer
    joblib.dump(vectorizer, vectorizer_filename)

    print('finishing svm')
    return svm_model

'''
    Load and return trained model from file 'filename'
'''
def load_trained_svm_model(filename):
    return joblib.load(filename)

'''
    Clean text:
    1. Remove links
    2. Readify text (clean, remove stopwords, stem)
    3. Replace covid-19/coronavirus with 'Corona', as hebrew speakers write 
'''
def clean_text(txt):
    txt = str(txt)
    ret = ' '.join(item for item in txt.split() if ((not (item.startswith('https://'))) and (not '.com' in item)))
    ret = readify_text(ret)
    ret = ret.replace('covid19', 'corona')
    ret = ret.replace('coronavirus', 'corona')
    ret = ret.replace('coronavir', 'corona')
    return ret

'''
    Clean text, for Hebrew data:
    1. Remove links
    2. Remove punctuation
'''
def clean_heb_text(txt):
    txt = str(txt)
    ret = ' '.join(item for item in txt.split() if ((not (item.startswith('https://'))) and (not '.com' in item)))
    ret = remove_punc(ret)
    return ret

'''
    Given a dataframe with Text column, clean text, and if it's Hebrew, extract keywords.
    Both are in additional columns
'''
def clean_dataset(df, heb=False):
    if heb:
        df['clean_text'] = df.apply(lambda row: clean_heb_text(row['text']), axis=1)
        df['keywords'] = df.apply(lambda row: get_keyWords(row['clean_text']), axis=1)
    else:
        df['clean_text'] = df.apply(lambda row: clean_text(row['Text']), axis=1)

def csv_cleaner(file, heb=False):
    df = pd.read_csv(file)
    df = df.drop_duplicates()
    df = df[df['binary label'].notna()] #we don't want to waste time cleaning unusable rows
    clean_dataset(df, heb)
    new_name = file.rsplit(".", 1)[0] + 'Clean.csv'
    df.to_csv(new_name, encoding='utf-8', index=False, mode='w+')

def clean_combined_eng_datasets():
    csv_cleaner('trueNews.csv')
    csv_cleaner('fakeNews.csv')
    csv_cleaner('Constraint_Train.csv')
    csv_cleaner('Constraint_Val.csv')
    csv_cleaner('english_test_with_labels.csv')
    csv_cleaner('corona_fake.csv')

def training_combined_eng_datasets():
    #********************** COMBINED DAASETS **********************************
    df_true = pd.read_csv('trueNewsClean.csv')
    df_false = pd.read_csv('fakeNewsClean.csv')
    df_Constraint = pd.read_csv('Constraint_TrainClean.csv')
    df_ConstraintVal = pd.read_csv('Constraint_ValClean.csv')
    df_en_test = pd.read_csv('english_test_with_labelsClean.csv')
    df_corona_fake = pd.read_csv('corona_fakeClean.csv')
    df_true.dropna(inplace=True)
    df_false.dropna(inplace=True)
    df_Constraint.dropna(inplace=True)
    df_ConstraintVal.dropna(inplace=True)
    df_en_test.dropna(inplace=True)
    df_corona_fake.dropna(inplace=True)
    frames = [df_true, df_false, df_Constraint, df_ConstraintVal, df_en_test, df_corona_fake]
    df = pd.concat(frames, join='inner')
    df.drop_duplicates()
    labels = df['Binary Label']
    #***************************************************************************
    text_train, text_valid, label_train, label_valid = train_test_split(df['clean_text'], labels, test_size=0.2,
                                                                        random_state=109)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', ngram_range=(1, 2), norm=None)
    tfidf_train = tfidf_vectorizer.fit_transform(text_train)
    tfidf_valid = tfidf_vectorizer.transform(text_valid)
    svm_classifier = our_svm(tfidf_train, label_train, tfidf_valid, label_valid, 'combined_trained_model.pkl', tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    #clf_classifier = our_clf(tfidf_train, label_train, tfidf_valid, label_valid, 'CLF_combined_trained_model.pkl', tfidf_vectorizer, 'CLF_tfidf_vectorizer.pkl')
    our_svm_tests(tfidf_vectorizer, svm_classifier)

#==========================HEBREW TRAINING=========================================================
'''
    Training for Hebrew dataset. with helper function
    all the changes of the new word2vec in gensim: https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
'''

def from_str_to_lst(text):
    text = text.replace("\'", "")
    text = text.replace("]", "")
    text = text.replace("[", "")
    arr = text.split(', ')
    return arr

def get_weight_matrix(model, vocab_size, vocab, DIM=100):
    weight_matrix = np.zeros((vocab_size, DIM))
    for word,i in vocab.items():
        weight_matrix[i] = model.wv[word]
    return weight_matrix

class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('onlplab/alephbert-base') #our BERT is AlephBERT
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)     #this means that the network is dense (fully connected - every neuron from dropout layer connects to every neuron in this layer). 768 neurons in this Linear layer, outputs to a single neuron
        self.sigmoid = nn.Sigmoid()         #sigmoid is not a layer, but a function. normalizes the output from the neural network

    def forward(self, tokens, masks=None):
        pooled_output = self.bert(tokens, attention_mask=masks)['pooler_output']
        dropout_output = self.dropout(pooled_output)    #the dropout layer randomly drops some of BERT's previous layers
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba

def training_heb(filename, model_filename, tokenizer_filename):
    df = pd.read_csv(filename)


    #==============================W2V==============================================================
    # df['keywords'] = df.apply(lambda row: from_str_to_lst(row['keywords']), axis=1)
    # X = df['keywords']  # X is a list of keyword-lists
    # DIM = 100
    # w2v_model = gensim.models.Word2Vec(sentences=X,  vector_size=DIM, window=10, min_count=1)
    # print(len(w2v_model.wv)) #this is for us, see how many words came from the model
    # #TRAINING
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(X)           #fit tokenizer for our vocabulary (taken from keywords)
    # X = tokenizer.texts_to_sequences(X) #convert vectors to sequences
    # maxlen = 50 #after checking the histogram of the length of all items in X
    # X = pad_sequences(X, maxlen=maxlen)
    # vocab_size = len(tokenizer.word_index) + 1
    # embedding_vectors = get_weight_matrix(w2v_model, vocab_size, tokenizer.word_index, DIM=DIM)
    # model = Sequential()
    # model.add(Embedding(vocab_size, output_dim=DIM, weights=[embedding_vectors], input_length=maxlen, trainable=False))
    # model.add(LSTM(units=8, activation = 'sigmoid', kernel_regularizer='l2'))
    # model.add(Dense(1, activation = 'sigmoid'))
    # opt = Adam(learning_rate=0.001)
    # model.compile(loss='binary_crossentropy', optimizer=opt)
    # print("model summary:\n")
    # print(model.summary())
    # labels = df['binary label']
    # X_train, X_test, y_train, y_test = train_test_split(X, labels, stratify=labels)
    # model.fit(X_train, y_train, validation_split=0.3, epochs=32)
    # #TESTING
    # y_pred = (model.predict(X_test)>=0.5).astype(int)
    # print(accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    #==============================W2V==============================================================


    #================ALEPHBERT===========================================================
    #were doing transfer learning, because this model already learned a lot
    BATCH_SIZE = 16      #try 16. or 32. training will be faster.
                        #after BATCHSIZE samples, will update the network
    EPOCHS = 50          #try changing to 50. if after a certaing number of epochs there isnt a drastic change, lower the number
    #leave learning rate small, because we dont want to drastically change everything bert learned before
    LEARNING_RATE = 3e-6   #tried 0.001, 0.0001, 3e-6. all pretty much the same acc result

    df['keywords'] = df.apply(lambda row: from_str_to_lst(row['keywords']), axis=1)
    X = df['keywords']
    # X = df['clean_text'] #commented out when trying keywords
    labels = df['binary label']
    alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    # alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
    #
    # # if not finetuning - disable dropout
    # alephbert.eval()
    X_train, X_test, y_train, y_test = train_test_split(X, labels, stratify=labels)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # train_tokens = list(map(lambda t: ['[CLS]'] + alephbert_tokenizer.tokenize(t)[:511], X_train))
    # test_tokens = list(map(lambda t: ['[CLS]'] + alephbert_tokenizer.tokenize(t)[:511], X_test))
    train_tokens = list(map(lambda t: ['[CLS]'] + t[:511], X_train)) #when using keywords
    test_tokens = list(map(lambda t: ['[CLS]'] + t[:511], X_test))   #when using keywords
    train_tokens_ids = list(map(alephbert_tokenizer.convert_tokens_to_ids, train_tokens))
    test_tokens_ids = list(map(alephbert_tokenizer.convert_tokens_to_ids, test_tokens))

    maxlen = 250         #250 is YAP's limit, 512 is BERT's limit
    train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=512)
    test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=512)

    #generate training and testing masks
    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
    test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]
    train_masks_tensor = torch.tensor(train_masks)
    test_masks_tensor = torch.tensor(test_masks)
    #Generate token tensors for training and testing
    train_tokens_tensor = torch.tensor(train_tokens_ids)
    train_y_tensor = torch.tensor(y_train.reshape(-1, 1)).float()
    test_tokens_tensor = torch.tensor(test_tokens_ids)
    test_y_tensor = torch.tensor(y_test.reshape(-1, 1)).float()
    #prepare our data loaders
    train_dataset = torch.utils.data.TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_dataset = torch.utils.data.TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)


    bert_clf = BertBinaryClassifier()
    optimizer = torch.optim.Adam(bert_clf.parameters(), lr=LEARNING_RATE)

    for epoch_num in range(EPOCHS):
        bert_clf.train()
        train_loss = 0
        for step_num, batch_data in enumerate(train_dataloader):
            token_ids, masks, labels = tuple(t for t in batch_data)
            probas = bert_clf(token_ids, masks)
            loss_func = nn.BCELoss()
            batch_loss = loss_func(probas, labels)
            train_loss += batch_loss.item()
            bert_clf.zero_grad()
            batch_loss.backward()
            optimizer.step()
            print('Epoch: ', epoch_num + 1)
            print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(X_train) / BATCH_SIZE, train_loss / (step_num + 1)))

    #evaluate
    bert_clf.eval()
    bert_predicted = []
    all_logits = []
    with torch.no_grad():
        for step_num, batch_data in enumerate(test_dataloader):
            token_ids, masks, labels = tuple(t for t in batch_data)
            logits = bert_clf(token_ids, masks)
            loss_func = nn.BCELoss()
            loss = loss_func(logits, labels)
            numpy_logits = logits.cpu().detach().numpy()

            bert_predicted += list(numpy_logits[:, 0] > 0.5)
            all_logits += list(numpy_logits[:, 0])


    #save trained model
    joblib.dump(bert_clf, model_filename)

    #save tokenizer
    joblib.dump(alephbert_tokenizer, tokenizer_filename)

    print(classification_report(y_test, bert_predicted))
    #================ALEPHBERT===========================================================
def pad(post, size):
    return [0]*abs(len(post)-size) + post


def grade_single_post(post, model, tokenizer):
    #load trained model and fitted tokenizer
    bert_clf = joblib.load(model)
    tokenizer = joblib.load(tokenizer)
    #prepare post
    post = clean_heb_text(post)
    post = get_keyWords(post)
    post = ['[CLS]'] + post[:511]
    #AlephBERT post
    post_ids = tokenizer.convert_tokens_to_ids(post)
    post_ids = pad(post_ids, 512)
    # post_ids = nn.utils.rnn.pad_sequence(post_ids, batch_first='pre', padding_value=0)

    #generate masks
    post_masks = [float(i > 0) for i in post_ids]
    post_masks_tensor = torch.tensor([post_masks])
    post_tokens_tensor = torch.tensor([post_ids])

    #evaluate
    bert_clf.eval()
    with torch.no_grad():
        # token_ids, masks, labels = tuple(t for t in batch_data)
        logits = bert_clf(post_tokens_tensor, post_masks_tensor)
        numpy_logits = logits.cpu().detach().numpy()

        bert_predicted = (numpy_logits[:, 0] > 0.5)
        all_logits = (numpy_logits[:, 0])

    return float(bert_predicted)




    #confusion matrix
    # conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # ax.matshow(conf_matrix)
    # plt.xlabel('Predictions')
    # plt.ylabel('Actuals')
    # plt.show()
#======================================================================================================


if __name__ == '__main__':
    #clean_combined_eng_datasets()  # uncomment when datasets change, or when cleaning process changes
    # training_combined_eng_datasets()  #uncomment when we want to redo training
    # *************** MANUAL CHECKS **********************
    # svm_model = 'combined_trained_model.pkl'
    # vectorizer = 'tfidf_vectorizer.pkl'
    # grade_single_post('"קבלו רמז: אני לא התחסנתי בכלל ולא נדבקתי ולא הייתי חולה. ביי "', svm_model, vectorizer)
    # grade_single_post("חבל כל חיסון מגביר את הסיכוי להידבק עוד הפעם", svm_model, vectorizer)
    # grade_single_post("שאלה בורה, החיסון ידוע לכל אחד שלא מונע הדבקה", svm_model, vectorizer)
    # grade_single_post("השגרירות ההודית בטוקיו אמרה שיותר מחבר צוות הודי אחד על נסיכת היהלום נמצא חיובי לקורונה", svm_model, vectorizer)
    # grade_single_post("Just in: Novel coronavirus named 'Corona': UN health agency. (AFP)", svm_model, vectorizer)
    # grade_single_post("WHO officially names coronavirus as Corona. CoronaOutbreak", svm_model, vectorizer)
    # grade_single_post("The Indian Embassy in Tokyo has said that one more Indian crew member on Diamond Princess has tested positive for Corona.", svm_model, vectorizer)
    # grade_single_post("קורונה גורמת לתחלואה קשה ותמותה גם בקרב צעירים שאינם מחוסנים 📈 עד גיל 50 - מי שלא התחסנו כלל, מהווים 69% מהחולים קשה ו-96% מהנפטרים (בין התאריכים 1.6 ל-4.9). אלו העובדות. חיסון ראשון, שני או שלישי - פשוט צאו להתחסן!", svm_model, vectorizer)
    # grade_single_post("קורונה זו לא רק 'מחלה של מבוגרים'! גם צעירים שלא מתחסנים חושפים עצמם למחלה קשה. 85% ממאושפזי הקורונה אשר נמצאים כעת במצב קריטי ומחוברים למכשיר אקמו - אינם מחוסנים. גילם הממוצע - 47. קורונה עלולה להיות מחלה קשה למבוגרים ולצעירים אבל בעיקר ללא מחוסנים. צאו להתחסן.", svm_model, vectorizer)
    # grade_single_post("עשרות, מאות, וכנראה אלפי אנשים שהוזרקו, מתים סובלים מדלקות בלב מנכויות קשות, מדום לב חולים במחלה שהתחסנו ממנה הכל על פי עדויות ממקור ראשון שנאספות בקושי ומראות רק את קצה הקרחון בעוד שהתמונה האמיתית נשארת במחשכים, מצונזרת ומוסתרת באלימות פראית ", svm_model, vectorizer)

    #***********HEB DATA!******************************
    # filename = 'NEW_manual_data_our_tags - NEW_manual_data.csv'
    # # csv_cleaner(filename, heb=True)
    # # filename = 'dataset_for_code_testing - Sheet1.csv'
    # # csv_cleaner(filename, heb=True)
    # clean_filename = filename.rsplit(".", 1)[0] + 'Clean.csv'
    # training_heb(clean_filename, model_filename='BERT_model.pkl', tokenizer_filename='AlephBERT_tokenizer.pkl')
    model = 'BERT_model.pkl'
    tokenizer = 'AlephBERT_tokenizer.pkl'
    print(grade_single_post("חבל כל חיסון מגביר את הסיכוי להידבק עוד הפעם", model, tokenizer))


"""
optional classifiers to add:
1. random forest classifier (from sklearn)
2. Multinomial Naive Bayes Algorithm (need to read more about it, we're not sure how it fits in). (from: https://www.analyticsvidhya.com/blog/2021/06/build-your-own-fake-news-classifier-with-nlp/)
3. we have 5 different classifier examples in the article:
https://www.researchgate.net/profile/Marten-Risius/publication/326405790_Automatic_Detection_of_Fake_News_on_Social_Media_Platforms/links/5b7df935a6fdcc5f8b5de39c/Automatic-Detection-of-Fake-News-on-Social-Media-Platforms.pdf
(page 11)
we want to create different functions for each classifier, and somehow work with the different results to get a more accurate analysis
"""

'''
pages talking about covid and vaccines:
FAKE:
1. mor sagmon: https://www.facebook.com/mor.sagmon
2. https://www.facebook.com/groups/VaccineChoiceIL/
3. https://www.facebook.com/groups/173406684888542/

REAL:
1. https://www.facebook.com/groups/440665513171433/about
'''