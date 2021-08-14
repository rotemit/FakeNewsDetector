import csv

from flask import Flask, request
from flask_cors import CORS

from machine_learning.scrap_posts import scrap_posts
import pandas as pd

# try:
#     from machine_learning import scrap_posts
# except ImportError:
#     from ...machine_learning import scrap_posts

app = Flask(__name__)
CORS(app)

def get_posts(name):
    post = "data from server " + name
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
    return post


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

if __name__ == '__main__':
    app.run(debug=True)
