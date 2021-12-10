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

"""
Citation:
  title={AlephBERT: a Pre-trained Language Model to Start Off your Hebrew NLP Application}, 
  
  author={Amit Seker, Elron Bandel, Dan Bareket, Idan Brusilovsky, Shaked Refael Greenfeld, Reut Tsarfaty},
  
  year={2021}
"""


"""
    Remove punctuation
"""
def remove_punc(txt):
    #make lowercase
    txt = txt.lower()
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~–'''
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
    txt = txt.replace('dna', 'דנא')
    txt = txt.replace('pims', 'פימס')
    txt = txt.replace('dna', 'דנא')
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
    Create a new csv file from a given csv file with data.
    Drop duplicates, remove rows with null labels, insert a new clean_text column with clean text
'''
def csv_cleaner(file):
    df = pd.read_csv(file)
    df = df.drop_duplicates()
    df = df[df['binary label'].notna()] #we don't want to waste time cleaning unusable rows
    df['clean_text'] = df.apply(lambda row: clean_heb_text(row['text']), axis=1)
    new_name = file.rsplit(".", 1)[0] + 'Clean.csv'
    df.to_csv(new_name, encoding='utf-8', index=False, mode='w+')


#**********************DATA AUGMENTATION*********************
'''
   Two functions for performing Random swap.
   swaps n pairs of words in a sentence
'''
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

'''
    Randomly delete words in a sentence; delete a word with probability p
'''
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

'''
    Main function for training data augmentation.
    Gets the training sentences and labels, and size of training data.
    Returns new data, with augmented sentences
'''
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

    return merged['sentence'], merged['label']


#*****************AlephBERT Wrapper*********************************************
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

#**********************Hebrew Training********************************
'''
    Run training for Hebrew data. Save trained model in model_filename, and tokenizer in tokenizer_filename,
    for future evaluation.
    We invite the user to play with BATCH_SIZE and EPOCHS, and observe the different results.
'''
def training_heb(filename, model_filename, tokenizer_filename):
    df = pd.read_csv(filename)
    BATCH_SIZE = 16
    EPOCHS = 6
    LEARNING_RATE = 3e-6

    X = df['clean_text']
    labels = df['binary label']
    alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')

    X_train, X_test, y_train, y_test = train_test_split(X, labels, stratify=labels, train_size=0.85)

    #augment training set to add more data and avoid overfitting
    X_train, y_train = augment_training_data(X_train, y_train, labels.size)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    train_tokens = list(map(lambda t: ['[CLS]'] + alephbert_tokenizer.tokenize(t)[:511], X_train))
    test_tokens = list(map(lambda t: ['[CLS]'] + alephbert_tokenizer.tokenize(t)[:511], X_test))
    train_tokens_ids = list(map(alephbert_tokenizer.convert_tokens_to_ids, train_tokens))
    test_tokens_ids = list(map(alephbert_tokenizer.convert_tokens_to_ids, test_tokens))

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

#****************************Functions for grading a single post*************
def pad(post, size):
    return [0]*abs(len(post)-size) + post

'''
    For a single post, prepare it for evaluation: clean, tokenize using tokenizer, generate masks.
    Evaluate using saved BERT model. Return a floating point number between 0-1, 0 being fake and 1 real
'''
def grade_single_post(post, model, tokenizer):
    #load trained model and fitted tokenizer
    bert_clf = joblib.load(model)
    tokenizer = joblib.load(tokenizer)
    #prepare post
    post = clean_heb_text(post)
    post = post.split()
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

#=========================================MAIN=============================================================

'''
    Run this main function if you wish to run training.
    If your csv file is new, run cleaning first.
    your trained model will be saved in model_filename, and the tokenizer in tokenizer_filename
'''
if __name__ == '__main__':
    filename = 'heb_data.csv'
    # csv_cleaner(filename)
    clean_filename = filename.rsplit(".", 1)[0] + 'Clean.csv'
    training_heb(clean_filename, model_filename='BERT_model.pkl', tokenizer_filename='AlephBERT_tokenizer.pkl')