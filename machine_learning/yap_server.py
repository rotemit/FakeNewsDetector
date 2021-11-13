import time

import requests
import random


'''
    Words with these tags are in key positions.
    according to:
    https://www.sketchengine.eu/hebrew-yap-part-of-speech-tagset/
'''
key_pos_tags = ['BN', 'BNN', 'NN', 'NN_S_PP', 'NNP', 'NNT', 'VB', 'VB_TOINFINITIVE']

"""
    Given a string, return a list of its lemmatized keywords
"""
def get_keyWords(text):
    text = text.replace(r'"', r'\"')
    url = 'https://www.langndata.com/api/heb_parser?token=2d3c69f56dbf39c1687ba1cf81162adb'
    _json = '{"data":"' + text + '"}'
    r = requests.post(url, data=_json.encode('utf-8'), headers={'Content-type': 'application/json; charset=utf-8'})
    #check each word in text. if it's in a key position, get lemma
    lemmas = []
    # tokens = r.json()['md_lattice']
    tokens = r.json()['dep_tree']
    #from checking only the word לא has neg dependency part, so it only add it
    for i in tokens:
        # if tokens[i].get('pos') in key_pos_tags:
        if tokens[i].get('pos') in key_pos_tags or "neg" in tokens[i].get('dependency_part'): #so we could add the word לא
            lemmas.append(tokens[i].get('lemma'))
    print(lemmas)
    time.sleep(3)
    return lemmas


"""
    Given a string, return a list of its lemmatized words
"""
def get_lemmas(text):
    text = text.replace(r'"', r'\"')
    url = 'https://www.langndata.com/api/heb_parser?token=2d3c69f56dbf39c1687ba1cf81162adb'
    _json = '{"data":"' + text + '"}'
    r = requests.post(url, data=_json.encode('utf-8'), headers={'Content-type': 'application/json; charset=utf-8'})
    #check each word in text. if it's in a key position, get lemma
    lemmas = []
    # tokens = r.json()['md_lattice']
    tokens = r.json()['dep_tree']
    #from checking only the word לא has neg dependency part, so it only add it
    for i in tokens:
        lem = tokens[i].get('lemma')
        # lemmas.append(tokens[i].get('lemma'))
        if len(lem) > 1:
            lemmas.append(lem)
    print(lemmas)
    time.sleep(3)
    return lemmas

def random_lemmas(words, p):
    print('words: '+words)
    text = words.replace(r'"', r'\"')
    url = 'https://www.langndata.com/api/heb_parser?token=2d3c69f56dbf39c1687ba1cf81162adb'
    _json = '{"data":"' + text + '"}'
    r = requests.post(url, data=_json.encode('utf-8'), headers={'Content-type': 'application/json; charset=utf-8'})

    tokens = r.json()['dep_tree']


    new_words = []
    for i in tokens:
        r = random.uniform(0, 1)
        word = tokens[i]
        if r > p:
            new_words.append(word.get('word'))
        else:
            new_words.append(word.get('lemma'))

    sentence = ' '.join(new_words)
    print('sentence: '+sentence)
    time.sleep(3)
    return sentence

if __name__ == '__main__':
    get_keyWords('אלו הרבה מילים שאני רושמת כאן לדוגמא הזו, אבל אני לא מבינה ואינני מבינה גם כן')