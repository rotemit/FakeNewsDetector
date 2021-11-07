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

def get_trust_value(url):
    return 0.7

def get_machine_value(url):
    return 1

def get_semantic_value (url):
    return 0.5

def get_posts(name):
    # post = "data from server " + name
    file = open('file.csv', 'w', encoding='UTF8')
    writer = csv.writer(file)
    writer.writerow(['post_id', 'post_text', 'date', 'writer', 'label'])
    scrap_posts("account", name, writer, 10)
    # writer.writerow(['1', name, 'date', 'writer', 'label'])
    # writer.writerow(['1', name, 'date', 'writer', 'label'])
    # writer.writerow(['1', name, 'date', 'writer', 'label'])
    # writer.writerow(['1', name, 'date', 'writer', 'label'])
    # writer.writerow(['1', name, 'date', 'writer', 'label'])
    file.close()
    print("done")
    return True

def scrap_post(text):
    driver = init_sel() #to init the driver
    if not login(driver, "ofrishani10@walla.com", "Ls5035"):
        login(driver, "ofrishani10@walla.com", "Is5035") #login in - return true on success, false otherwise.

    print("start")
    global post
    account = scrap_url(driver, text, posts=5, loging_in=True)
    analyzed = analyze_facebook(account)
    print("done")
    # https://www.facebook.com/ofri.shani.31/posts/10216864802065081
    return (vars(analyzed))

# Decorator defines a route
# http://localhost:5000/
@app.route('/', methods=['GET'])
@app.route('/profile', methods=['POST'])
def get_query_from_react():
    data = request.get_json()
    # get_posts(data)
    return get_posts(data)


@app.route('/profile', methods=['GET'])
def send_data_to_react():
    df = pd.read_csv('file.csv')
    print(df['post_text'])
    posts = []
    for post in df['post_text']:
        posts.append(post)
    return {
        'name': posts
    }
@app.route('/scanPost', methods=['POST'])
def get_post():
    data = request.get_json()
    # get_posts(data)
    # print(data)
    return scrap_post(data)

@app.route('/login', methods=['POST'])
def get_login_details():
    driver = init_sel()
    data = request.get_json()
    # get_posts(data)
    print(data)
    return login(driver, data.name, data.password)


# @app.route('/scanPost', methods=['GET'])
# def send_post_result():
#     return "1"

@app.route('/scanPost', methods=['GET'])
def send_post_result():
    # post = json.load(f)
    # print(post["content"])
    print(get_semantic_value("aaa"))
    return {
        'trust_value': str(get_trust_value("aaa")),
        'machine_value': str(get_machine_value("aaa")),
        'semantic_value': str(get_semantic_value("aaa")),
    }
if __name__ == '__main__':
    app.run(debug=True)
