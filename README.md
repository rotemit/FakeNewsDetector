# FakeNewsDetector

Hello! Welcome to our FakeNewsDetector, the ultimate tool for Fake News Analysis in Hebrew Facebook posts regarding Covid-19. This analyzer was created for our undergraduate project in computer science.

First, make sure you have Nodejs, if not download it and add its PATH to your environment variable.

Follow these steps to run the project:
1. Download project:
```ruby
#Clone the project
> git clone https://github.com/rotemit/FakeNewsDetector.git
#cd into the project folder
> cd FakeNewsDetector
```
2. Create and activate virtual environment:
```ruby
#Create
> py -3 -m venv fakeNewsVenv
#Activate
> fakeNewsVenv\Scripts\activate
```
3. Install requirements:
```ruby
(fakeNewsVenv) > pip install -r requirements.txt
```
4. Download the trained ML model file 'BERT_model.pkl' from this link:<\br>
https://drive.google.com/drive/folders/1XjgxUfRcZhu_Mr_T_TmUwZGMP9XTmz6q?usp=sharing
<\br>Move the file to the folder: FakeNewsDetector/machine_learning

Next, run the server:
```
(fakeNewsVenv) > python api.py
```


Finally, open a new command line window and use it to run the client:
```ruby
#from the main project folder, cd into my-app. (Without activated env)
> cd my-app
> npm install
> set NODE_OPTIONS=--openssl-legacy-provider
> npm start
```
*npm install and set NODE_OPTIONS  should only be done the first time you run the program

Open your browser, and navigate to URL:
https://localhost:3000

Follow the instructions on the page to detect some Hebrew Fake News :)

