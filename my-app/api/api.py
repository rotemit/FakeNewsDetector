import csv

from flask import Flask, request
import scrap_posts
import pandas as pd

app = Flask(__name__)


def get_posts(name):
    post = "data from server " + name;
    file = open('file.csv', 'w', encoding='UTF8')
    writer = csv.writer(file)
    writer.writerow(['post_id', 'post_text', 'date', 'writer', 'label'])
    scrap_posts("account", name, writer, 10)
    file.close()
    print("done")
    return post;


# Decorator defines a route
# http://localhost:5000/
@app.route('/', methods=['GET'])
@app.route('/profile', methods=['POST'])
def get_query_from_react():
    data = request.get_json()
    get_posts(data)
    return data


@app.route('/profile', methods=['GET'])
def send_data_to_react():
    df = pd.read_csv('file.csv')
    print(df['post_text'])
    for post in df['post_text']:
        return {
            'name': post
        }

if __name__ == '__main__':
    app.run(debug=True)
