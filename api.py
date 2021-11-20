import csv
import json

from flask import Flask, request
from flask_cors import CORS

from Scraper.scrapper import scrap_url, init_sel, login
from Analyzer.Analyzer import analyze_facebook, analyze_string
from machine_learning.datatrain import BertBinaryClassifier

import pandas as pd


# try:
#     from machine_learning import scrap_posts
# except ImportError:
#     from ...machine_learning import scrap_posts

app = Flask(__name__)
CORS(app)
post = None

userName = ''
password=''
driver = init_sel()  # to init the driver

def scrap_post(text, numOfPosts):
    print(userName)
    print(password)

    first_chars = text[0:4]
    if first_chars!='http':
        return vars(analyze_string(text))
    global post, driver
    account = scrap_url(driver, text, posts=int(numOfPosts), loging_in=True)
    if isinstance(account, str):
        return {'error': account }

    analyzed = analyze_facebook(account)
    return (vars(analyzed))

@app.route('/scanPost', methods=['POST'])
def get_post():
    data = request.get_json()
    return scrap_post(data['url'], data['numOfPosts'])

@app.route('/login', methods=['POST'])
def get_login_details():
    data = request.get_json()
    global userName, password
    userName =  data['name']
    password =  data['password']
    global driver
    return {'result': (login(driver, userName, password))}

if __name__ == '__main__':
    app.run(debug=True)
