import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from machine_learning.stop_words import stop_words
from machine_learning.yap_server import get_keyWords, get_lemmas, random_lemmas
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim
from transformers import BertModel, BertTokenizerFast
import torch.nn as nn
import torch
import random

#pips: pip install tensorflow --user
#       pip install gensim
#       pip install transformers    #for AlephBERT
#       pip install torch torchvision torchaudio    #for AlephBERT



"""
    Remove punctuation
"""
def remove_punc(txt):
    #make lowercase
    txt = txt.lower()
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~–'''

    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in txt:
        if ele in punc:
            txt = txt.replace(ele, "")
    return txt


'''
    Replace recurring English words liks 'fda' with corresponding Hebrew words
'''
def replace_recurring_english_words(txt):
    txt = txt.lower()
    txt = txt.replace('mrna', 'מרנא')
    txt = txt.replace('rna', 'רנא')
    txt = txt.replace('fda', 'פדא')
    txt = txt.replace('pims', 'פימס')
    txt = txt.replace('see more', '')
    return txt


'''
    Clean text, for Hebrew data:
    1. Remove links
    2. Remove punctuation
'''
def clean_heb_text(txt):
    txt = str(txt)
    ret = ' '.join(item for item in txt.split() if ((not (item.startswith('https://'))) and (not '.com' in item)))
    ret = remove_punc(ret)
    ret = replace_recurring_english_words(ret)
    return ret

'''
    Given a dataframe with Text column, clean text, and if it's Hebrew, extract keywords.
    Both are in additional columns
'''
def clean_dataset(df, heb=False):
    if heb:
        df['clean_text'] = df.apply(lambda row: clean_heb_text(row['text']), axis=1)
        df['lemmatized'] = df.apply(lambda row: get_lemmas(row['clean_text']), axis=1)
        # df['keywords'] = df.apply(lambda row: get_keyWords(row['clean_text']), axis=1)
    else:
        df['clean_text'] = df.apply(lambda row: clean_text(row['Text']), axis=1)

def csv_cleaner(file, heb=False):
    df = pd.read_csv(file)
    df = df.drop_duplicates()
    df = df[df['binary label'].notna()] #we don't want to waste time cleaning unusable rows
    clean_dataset(df, heb)
    new_name = file.rsplit(".", 1)[0] + 'Clean.csv'
    df.to_csv(new_name, encoding='utf-8', index=False, mode='w+')



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


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1

        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_swap(words, n):
    words = words.split()
    new_words = words.copy()

    for _ in range(n):
        new_words = swap_word(new_words)

    sentence = ' '.join(new_words)

    return sentence


def random_deletion(words, p):
    words = words.split()

    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    sentence = ' '.join(new_words)

    return sentence


def augment_training_data(X_train, y_train, size):
    merged = pd.DataFrame()
    merged['sentence'] = X_train
    merged['label'] = y_train

    #Random Swap Augmentation
    #randomly select rows
    to_augment = merged.sample(frac = 0.07)
    #augment these rows
    for _, row in to_augment.iterrows():
        augmented_sentence = random_swap(row['sentence'], 1)
        merged.loc[str(size)] = [augmented_sentence, row['label']]
        size += 1

    #Random Deletion Augmetation
    #randomly select rows
    to_augment = merged.sample(frac = 0.07)
    #augment these rows
    for _, row in to_augment.iterrows():
        augmented_sentence = random_deletion(row['sentence'], p=0.17)
        merged.loc[str(size)] = [augmented_sentence, row['label']]
        size += 1

    # #Random Lemmatization Augmentation
    # to_augment = merged.sample(frac=0.1)
    # # augment these rows
    # for _, row in to_augment.iterrows():
    #     augmented_sentence = random_lemmas(row['sentence'], p=0.2)
    #     merged.loc[str(size)] = [augmented_sentence, row['label']]
    #     size += 1
    #
    return merged['sentence'], merged['label']



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
    #================ALEPHBERT===========================================================
    #were doing transfer learning, because this model already learned a lot
    BATCH_SIZE = 16     #try 16. or 32. training will be faster.
                        #after BATCHSIZE samples, will update the network
    EPOCHS = 6          #try changing to 50. if after a certaing number of epochs there isnt a drastic change, lower the number
    #leave learning rate small, because we dont want to drastically change everything bert learned before
    LEARNING_RATE = 3e-6   #tried 0.001, 0.0001, 3e-6. all pretty much the same acc result

    # df['keywords'] = df.apply(lambda row: from_str_to_lst(row['lemmatized']), axis=1)
    # X = df['keywords']
    X = df['clean_text'] #commented out when trying keywords
    labels = df['binary label']
    alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    # alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
    #
    # # if not finetuning - disable dropout
    # alephbert.eval()
    X_train, X_test, y_train, y_test = train_test_split(X, labels, stratify=labels, train_size=0.85)

    #augment training set to add more data
    X_train, y_train = augment_training_data(X_train, y_train, labels.size)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    train_tokens = list(map(lambda t: ['[CLS]'] + alephbert_tokenizer.tokenize(t)[:511], X_train))
    test_tokens = list(map(lambda t: ['[CLS]'] + alephbert_tokenizer.tokenize(t)[:511], X_test))
    # train_tokens = list(map(lambda t: ['[CLS]'] + t[:511], X_train)) #when using keywords
    # test_tokens = list(map(lambda t: ['[CLS]'] + t[:511], X_test))   #when using keywords
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
    post = post.split()
    # print(post)
    # post = get_lemmas(post)
    # post = get_keyWords(post)
    post = ['[CLS]'] + post[:511]
    #AlephBERT post
    post_ids = tokenizer.convert_tokens_to_ids(post)
    post_ids = pad(post_ids, 512)

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
    to_retrun = numpy_logits[:, 0]
    print(to_retrun[0])
    return to_retrun[0]
    # return numpy_logits[:, 0]
    # return float(bert_predicted)

#======================================================================================================


if __name__ == '__main__':
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
    # filename = 'dataset_for_code_testing - Sheet1.csv'
    filename = 'NEW_manual_data_our_tags - NEW_manual_data.csv'
    # csv_cleaner(filename, heb=True)
    clean_filename = filename.rsplit(".", 1)[0] + 'Clean.csv'
    # training_heb(clean_filename, model_filename='BERT_model.pkl', tokenizer_filename='AlephBERT_tokenizer.pkl')
    training_heb(clean_filename, model_filename='BERT_model_clean_6_16_aug5.pkl', tokenizer_filename='AlephBERT_tokenizer.pkl')

    # filename = 'NEW_manual_data_our_tags - NEW_manual_data.csv'
    # clean_filename = filename.rsplit(".", 1)[0] + 'Clean.csv'
    # training_heb(clean_filename, model_filename='BERT_model_clean_20_16_aug2.pkl', tokenizer_filename='AlephBERT_tokenizer_clean_20_16_aug2.pkl')
    #
    # filename = 'NEW_manual_data_our_tags - NEW_manual_data.csv'
    # clean_filename = filename.rsplit(".", 1)[0] + 'Clean.csv'
    # training_heb(clean_filename, model_filename='BERT_model_clean_20_16_aug3.pkl',
    #              tokenizer_filename='AlephBERT_tokenizer_clean_20_16_aug3.pkl')

    # model = 'BERT_model_clean_6_16_aug.pkl'
    # tokenizer = 'AlephBERT_tokenizer_clean_6_16_aug.pkl'
    # # print(grade_single_post("יש לך בעיה רצינית בהבנת הנושא - החיסון ( לא ניסוי לא ניסוי ) עבר בדיקות מחמירות של רשות הבריאות האמריקאית , וסתם קצת השכלה יא דבע - זאת הרשות הכי מחמירה בבדיקות של חיסונים ותרופות . אל לך תלמד קצת אהבל !! ליפני שאתה מביע דיעות ללא שום בסיס מדעי", model, tokenizer))
    # # print(grade_single_post("ידוע לא מעט על פגיעה בבלוטת התריס ובלוטות הלימפה שליד מיד אחרי החיסון הראשון או השני", model, tokenizer))
    # # print(grade_single_post("אין מחקר שאומר מה השפעות על פריון על ילדים שחלו בקורונה זה יתבהר בעוד שנים כאשר יגיעו לגיל המתאים. כרגע רק מתחילים מחקרים על השפעות קורונה על ילדים עיכוב התפתחות וצמיחה והשפעות נוספות זה מחקרים שהתוצאות יהיו עוד שנים", model, tokenizer))
    # # print(grade_single_post("כיום הקורונה אינה פוגעת רק באנשים עם מחלות רקע קשה . מה גם שמחלות רקע לא הכוונה לאדם שעומד למות .", model, tokenizer))
    # print(grade_single_post("אין כללים. כל גוף מגיב אחרת. זה קורונה.", model, tokenizer))
    # print(grade_single_post("לרוב אזרחי העולם [הקורונה] זה בדיוק מה שזה- שפעת!", model, tokenizer))
    # print(grade_single_post("מי שחלה מחוסן נגד הדבקות שניה", model, tokenizer))
    # print(grade_single_post("גם מי שלא חוסן [עובר את המחלה בקלות], מניסיון. זה שפעת", model, tokenizer))

    # first = backward_translate("חבל כל חיסון מגביר את הסיכוי להידבק עוד הפעם")
    # print(first)
    # for i in range(5):
    #     first = random_swap("חבל כל חיסון מגביר את הסיכוי להידבק עוד הפעם", n=2)
    #     print(first)




'''
pages talking about covid and vaccines:
FAKE:
1. mor sagmon: https://www.facebook.com/mor.sagmon
2. https://www.facebook.com/groups/VaccineChoiceIL/
3. https://www.facebook.com/groups/173406684888542/

REAL:
1. https://www.facebook.com/groups/440665513171433/about
'''