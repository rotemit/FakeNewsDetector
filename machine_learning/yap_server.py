import time

import requests
import json

'''
    Returns Lemmas for a single string
'''
def get_lemma(text):
    # text = 'שלום שלום לכן ילדות חמודות'
    # Escape double quotes in JSON.
    text = text.replace(r'"', r'\"')
    url = 'https://www.langndata.com/api/heb_parser?token=2d3c69f56dbf39c1687ba1cf81162adb'
    _json = '{"data":"' + text + '"}'
    headers = {'content-type': 'application/json'}
    r = requests.post(url, data=_json.encode('utf-8'), headers={'Content-type': 'application/json; charset=utf-8'})
    # with open('yap_analysis.json', 'w', encoding='UTF8') as outfile:
    #     json.dump(r.json(), outfile, indent=4, ensure_ascii=False)
    time.sleep(3)
    return r.json().get('lemmas')

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
    tokens = r.json()['md_lattice']
    for i in tokens:
        if tokens[i].get('pos') in key_pos_tags:
            lemmas.append(tokens[i].get('lemma'))
    print(lemmas)
    time.sleep(3)
    return lemmas

if __name__ == '__main__':
    get_keyWords('אלו הרבה מילים שאני רושמת כאן לדוגמא הזו')