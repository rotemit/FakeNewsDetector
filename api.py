import csv
import json

from flask import Flask, request
from flask_cors import CORS

from machine_learning.scrap_posts import scrap_posts
from Scraper.scrapper import scrap_facebook
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
    print("start")
    global post
    post = scrap_facebook(url_post=text, loging_in=True, user_mail="ofrishani10@walla.com", user_password="Is5035")
    print("done")
    # https://www.facebook.com/ofri.shani.31/posts/10216864802065081
    return True

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
