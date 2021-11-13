import csv
import json

from flask import Flask, request
from flask_cors import CORS

from machine_learning.scrap_posts import scrap_posts
from Scraper.scrapper import scrap_url, init_sel, login
from Analyzer.Analyzer import analyze_facebook
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

def scrap_post(text, numOfPosts):
    print(userName)
    print(password)
    driver = init_sel() #to init the driver
    if not login(driver, userName, password):
        login(driver, userName, password) #login in - return true on success, false otherwise.

    print("start")
    global post
    account = scrap_url(driver, text, posts=int(numOfPosts), loging_in=True)
    analyzed = analyze_facebook(account)
    print("done")
    # https://www.facebook.com/ofri.shani.31/posts/10216864802065081
    return (vars(analyzed))

# Decorator defines a route
# http://localhost:5000/
# @app.route('/', methods=['GET'])
# @app.route('/profile', methods=['POST'])
# def get_query_from_react():
#     data = request.get_json()
#     # get_posts(data)
#     return get_posts(data)

@app.route('/scanPost', methods=['POST'])
def get_post():
    data = request.get_json()
    # get_posts(data)
    print(data)
    return scrap_post(data['url'], data['numOfPosts'])

@app.route('/login', methods=['POST'])
def get_login_details():
    driver = init_sel()
    data = request.get_json()
    # get_posts(data)
    global userName, password
    userName =  data['name']
    password =  data['password']
    print(data)
    return 'data'

if __name__ == '__main__':
    app.run(debug=True)
