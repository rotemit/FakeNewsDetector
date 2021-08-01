from flask import Flask, render_template

app = Flask(__name__)

# Decorator defines a route
# http://localhost:5000/
@app.route('/api', methods=['GET'])
def home():
    return {
        'name': 'Hello World'
    }

if __name__ == '__main__':
    app.run(debug=True)