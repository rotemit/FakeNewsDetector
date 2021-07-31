from flask import Flask

app = Flask('helloworld')

# Decorator defines a route
# http://localhost:5000/
@app.route('/')
def index():
    return "Hello World!"

if __name__ == '__main__':
    app.run()