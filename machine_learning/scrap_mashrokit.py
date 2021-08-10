from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import csv
grades = {"מטעה": 0, "נכון": 1, "נכון ברובו": 2, "חצי נכון": 3, "לא נכון ברובו": 4, "לא נכון": 5}


def gather_info(post):
    person_n_job = post.find_element_by_class_name("figure-position")
    person_n_job_arr = person_n_job.text.split('\n')
    person = person_n_job_arr[0]
    job = person_n_job_arr[1].replace(' • ', ' ')
    grading = post.find_element_by_class_name("grade-label")
    grade = grades.get(grading.text)
    themes = post.find_element_by_class_name("sub-title")
    theme = themes.text.replace(":", "")
    quote = post.find_element_by_class_name("main-quote")
    text = quote.text.replace("\"", "")
    return [person, job, grade, theme, text]


def scroll_down(driver, last_height):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    newHeight = driver.execute_script("return document.body.scrollHeight")
    return newHeight


def scrap(csv_writer, arr, count=500):
    driver = init_sel()
    driver.get("https://thewhistle.globes.co.il/feed")
    last_height = driver.execute_script("return document.body.scrollHeight")
    i=0
    # for i in range(pages):
    while i < count:
        new_height = scroll_down(driver, last_height)
        posts = driver.find_elements_by_class_name("card-wrapper")
        for post in posts:
            info = gather_info(post)
            if info not in arr:
                arr.insert(i, info)
                csv_writer.writerow(info)
                i += 1
                print(i)
        # if new_height == last_height:
        #     break
        last_height = new_height
    driver.quit()


def init_sel():
    PATH = "C:\Program Files (x86)\chromedriver.exe"
    driver = webdriver.Chrome(PATH)
    driver.maximize_window()
    return driver


if __name__ == "__main__":
    file = open('mashrokit.csv', 'w', encoding='UTF8')
    writer = csv.writer(file)
    writer.writerow(['person', 'job', 'label', 'theme', 'text'])
    arr = []
    scrap(writer, arr)
    file.close()
    print("done")
