# FakeNewsDetector

Hello! Welcome to our FakeNewsDetector, the ultimate tool for Fake News Analysis in Hebrew Facebook posts regarding Covid-19. This analyzer was created for our undergraduate project in computer science.

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
4. Download chromedriver version 95.0.4638.69:
https://chromedriver.storage.googleapis.com/index.html?path=95.0.4638.69/
place the downloaded .exe file in the main project directory (under FakeNewsDetector)


Next, run the server:
```
(fakeNewsVenv) > python api.py
``` 
*check if this needs to be inside venv


Finally, open a new command line window and use it to run the client:
```ruby
#from the main project folder, cd into my-app. (Without activated env)
> cd my-app
> npm install
> npm start
```
*npm install should only be done the first time you run the program

Open your browser, and navigate to URL:
https://localhost:3000

Follow the instructions on the page to detect some Hebrew Fake News :)

