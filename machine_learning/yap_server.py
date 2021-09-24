import time

import requests
import json

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