from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

from modules.Page import Page
from modules.Group import Group
from modules.Account import Account
from modules.Post import Post
import time
from datetime import date
from Analyzer.Analyzer import analyze_facebook
import pprint
import json
from machine_learning.datatrain import BertBinaryClassifier

"""
    some global needed variable, some might change in diffrent versions of Facebook
"""
is_logged_in = False
year_not_registered = "//*[contains(text(),'Born on ') or contains(text(),'No posts available')]"
friendship_duration_xpath = "//div[@class='rq0escxv l9j0dhe7 du4w35lb j83agx80 pfnyh3mw jifvfom9 gs1a9yip owycx6da btwxx1t3 discj3wi b5q2rw42 lq239pai mysgfdmx hddg9phg']"
# friendship_duration_xpath = "//span[@class='d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d3f4x2em fe6kdd0r mau55g9w c8b282yb iv3no6db jq4qci2q a3bd9o3v knj5qynh oo9gr5id hzawbc8m']"
name_of_page = "//div[@class='rq0escxv l9j0dhe7 du4w35lb j83agx80 cbu4d94t g5gj957u d2edcug0 p01isnhg rj1gh0hx dtpq6qua p8fzw8mz pcp91wgn ihqw7lf3 ipjc6fyt']"
name_of_group = "//div[@class='rq0escxv l9j0dhe7 du4w35lb j83agx80 cbu4d94t g5gj957u d2edcug0 hpfvmrgz on77hlbc buofh1pr o8rfisnq ph5uu5jm b3onmgus ihqw7lf3 ecm0bbzt']"
name_of_page_1 = "//span[@class='d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh a8c37x1j keod5gw0 nxhoafnm aigsh9s9 l1jc4y16 fe6kdd0r mau55g9w c8b282yb rwim8176 mhxlubs3 p5u9llcw hnhda86s oo9gr5id oqcyycmt']"
name_of_page_2 = "//span[@class='d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh a8c37x1j keod5gw0 nxhoafnm aigsh9s9 embtmqzv fe6kdd0r mau55g9w c8b282yb hrzyx87i m6dqt4wy h7mekvxk hnhda86s oo9gr5id hzawbc8m']"
about_group_fields = "//div[@class='dwo3fsh8 g5ia77u1 rt8b4zig n8ej3o3l agehan2d sk4xxmp2 rq0escxv q9uorilb kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x jb3vyjys rz4wbd8a qt6c0cv9 a8nywdso l9j0dhe7 i1ao9s8h k4urcfbm']"
to_english = "/?locale2=en_US"

class json_encoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__

"""
    initiation of the chrome driver
"""
def init_sel():
    options = webdriver.ChromeOptions()
    options.add_argument('headless') #so we will not see the open window
    PATH = "C:\Program Files (x86)\chromedriver.exe"
    driver = webdriver.Chrome(PATH, options=options)
    driver.maximize_window()
    return driver

def finish_sel(driver):
    driver.quit()
"""
    login in into Facebook with the giver email and password
    :return True if logged in succefully, else return False
"""
def login(driver, email, password):
    driver.get("https://www.facebook.com")
    search_email = driver.find_element_by_id("email")
    time.sleep(2)
    search_email.send_keys(email)
    search_pass = driver.find_element_by_id("pass")
    time.sleep(3)
    search_pass.send_keys(password)
    time.sleep(4)
    search_button = redirect_by_xpath(driver, "//div[@class='_6ltg']")
    if search_button is None:
        return False
    search_button[0].click()
    time.sleep(3)
    if "login" in driver.current_url:
        return False
    global is_logged_in
    is_logged_in = True
    return True


# not sure why we need this, but we'll see
def background_click(driver):
    time.sleep(0.5)
    actions = ActionChains(driver)
    actions.double_click()
    actions.perform()
    time.sleep(0.5)


"""
   This method serves to move between pages, it gets 3 arguments:
   the driver, string that describe the next page, and if initial click is needed (not sure why).
   the method, either works or return False. 
"""
def redirect(driver, next_page, initial_click=True):
    # if initial_click is true we preform a background click - not sure why
    if initial_click:
        background_click(driver)
    try:
        # we try to wait until the element we want to click to move to the next page will becaome clickable,
        # if after 7 second it's not clickable ( or simply doesn't exists) we return False;
        # if it is clickable, we click on it and move to the next page
        element = WebDriverWait(driver, 7).until(
            EC.element_to_be_clickable((By.LINK_TEXT, next_page))
        )
        element.click()
        time.sleep(2)
        if not is_logged_in:
            driver.get(driver.current_url+to_english)
    except:
        return False
    finally:
        print("redirected to " + next_page)


"""
    this method helps getting an element by its xpath, and it can wait until it's appear, unlike the built in method.
    can cause errors, if elements is not found after the waited time.
"""
def redirect_by_xpath(driver, xpath):
    time.sleep(1)
    try:
        sleep_time = 10
        if xpath == year_not_registered:
            sleep_time = 3
        element = WebDriverWait(driver, sleep_time).until(
            EC.presence_of_all_elements_located((By.XPATH, xpath))
        )
        return element
    except:
        return None
    finally:
        print(xpath)


"""
    this method extract the number of total friends of the page it is on
"""
def extract_total_friends(driver):
    if is_logged_in:
        total_friends = driver.find_element_by_partial_link_text('Friends').text
    else:
        menu = driver.find_elements_by_xpath("//div[@class='i09qtzwb rq0escxv n7fi1qx3 pmk7jnqg j9ispegn kr520xx4']/div")
        for elem in menu:
            if 'Friend' in elem.text:
                total_friends = elem.text
    temp = total_friends.partition("Friends\ufeff")
    if temp[2] == '':
        return -1
    return int(temp[2])


"""
    this method extract the profile summary of the page it is on,
    which consists of details the user chose to share, user's total friends, and the age of the user's account.
"""
def extract_profile_summary(driver):
    try:
        total_friends = extract_total_friends(driver)
    except:
        total_friends = 0
    if is_logged_in:
        age_of_account = get_age_of_account(driver)
    else:
        age_of_account = 0
    return total_friends, age_of_account

#scraping the "about" page
def scrap_about(driver):
    if is_logged_in:
        time.sleep(2)
        redirect(driver, "About")
    return extract_profile_summary(driver)


"""
    this method extract the friendship duration of the root user with the current page.
"""
def extract_friendship_duration(driver):
    if not is_logged_in:
        return 0
    upper_navigation_bar = redirect_by_xpath(driver, "//div[@class='ku2zlfd4 q3mryazl']/div/div")
    if upper_navigation_bar is None:
        return None
    time.sleep(2)
    try:
        upper_navigation_bar[len(upper_navigation_bar) - 1].click()
    except:
        return None
    isFriend = redirect(driver, 'See Friendship', False)
    if isFriend is False:
        return 0
    all_common_fields = redirect_by_xpath(driver, friendship_duration_xpath)
    if all_common_fields is None:
        return 0
    for field in all_common_fields:
        if field.text.startswith("Your friend since "):
            # strating from the 1st of the month, since not given exact day
            beginning_of_friendship = field.text.replace("Your friend since ", "1 ")
            return calculate_age(beginning_of_friendship)
    return 0

"""
    this method extract the the amount of mutual friend the user has with the account the driver is in.
"""
def extract_mutual_friends(driver, is_friend):
    all_fields = []
    searching_string = ""

    if is_friend:
        all_fields = redirect_by_xpath(driver, friendship_duration_xpath)
        searching_string = "mutual friends"
        if all_fields is None:
            return 0
    else:
        all_fields = driver.find_elements_by_xpath("//div[@class='j83agx80 btwxx1t3 bp9cbjyn jifvfom9']")
        searching_string = "Mutual Friends"

    for field in all_fields:
        if searching_string in field.text:
            mutual_friends_arr = field.text.split(' ')
            mutual_friends = int(mutual_friends_arr[0])
            return mutual_friends
    return 0


"""
    this method helps us to scroll to the bottom of the page,
    and its return the last element with the given xpath
"""
def scroll_to_bottom(driver, elements_xpath):
    # using the method of "send_keys" to type the END key to get to the end of the page.
    actions = ActionChains(driver)
    elements = driver.find_elements_by_xpath(elements_xpath)
    old_elements_amount = len(elements)
    while True:
        time.sleep(2)
        actions.send_keys(Keys.END)
        actions.perform()
        elements = driver.find_elements_by_xpath(elements_xpath)
        for elem in elements:
            print(elem.text)
        if len(elements) == old_elements_amount:
            break
        old_elements_amount = len(elements)
    return elements[len(elements) - 1].text


"""
    this method calculate the age of an account by substracting the date enrolled from today date
"""
def calculate_age(date_joined, short=False):
    date_arr = date_joined.split(' ')
    date_arr[1] = month_converter(date_arr[1], short)
    date_joined = date(int(date_arr[2]), date_arr[1], int(date_arr[0]))
    curr_date = date.today()
    number_of_days = curr_date - date_joined
    return number_of_days.days


# converter of string of month to number
def month_converter(month, short=False):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
    if short:
        months = ['Jan,', 'Feb,', 'Mar,', 'Apr,', 'May,', 'Jun,', 'Jul,',
                  'Aug,', 'Sep,', 'Oct,', 'Nov,', 'Dec,']
    return months.index(month) + 1


"""
    this method extracting the date of creation of an accout, which is not the root user.
    the method get the date by going over the posts of the user.
    it beggin in the middle, check if the string: "No posts available" or the string "Born on" is in there.
    it try to find the yearof creation by alwayes go in the middle until finding the year of creation.
"""
def get_age_of_account(driver):
    try:
        redirect(driver, "Posts")
    except:
        return 0
    mid = None
    left = 2008
    right = date.today().year
    try:
        while left < right:
            filter_button = redirect_by_xpath(driver, "//span[text()='Filters']")[0]
            filter_button.click()

            if mid is None:
                redirect_by_xpath(driver, "//span[text()='Year']")[0].click()
            else:
                redirect_by_xpath(driver, "//span[text()='" + str(mid) + "']")[0].click()
            mid = left + (right - left) // 2
            redirect_by_xpath(driver, "//span[text()='" + str(mid) + "']")[0].click()
            redirect_by_xpath(driver, "//span[text()='Done']")[0].click()
            time.sleep(1)
            try:
                redirect_by_xpath(driver, year_not_registered)
                left = mid + 1
            except:
                right = mid
    except:
        return 0
    # we assume the creation if January 1st of the found year
    return calculate_age("1 January " + str(left))

# change from date of type <MonthName DD, YYYY> to <DD MonthName YYYY>
def reformat_date(date):
    date_arr = date.replace(",", "").split(' ')
    return ' '.join([date_arr[1], date_arr[0], date_arr[2]])


"""
    This method extract the age of the page the driver is in.
    it uses the field of "Page Transparency" if exists
"""
def get_age_of_page(driver):
    elements = driver.find_elements_by_xpath("//div[@class='lpgh02oy']/div/div")
    for elem in elements:
        if "Page Transparency" in elem.text:
            trasp_arr = elem.text.split('\n')
            for trasp in trasp_arr:
                if trasp.startswith('Page created'):
                    date = trasp.replace("Page created - ", "")
                    new_date = reformat_date(date)
                    return calculate_age(new_date, False)

    redirect(driver, 'About')
    time.sleep(3)
    redirect(driver, "Page Transparency")
    time.sleep(2)
    fields = driver.find_elements_by_class_name("sjgh65i0")
    for field in fields:
        if "Creation Date" in field.text and 'About' not in field.text:
            date_arr = field.text.split('\n')
            new_date = reformat_date(date_arr[0])
            isHome = redirect(driver, "Home")
            if isHome is False:
                redirect(driver, "Posts")
            return calculate_age(new_date, False)
    return 0

# change from human number, such as 10K - to actual decimal number, such as 10000
def int_from_human_format(x):
    if type(x) == int:
        return x
    if 'K' in x:
        if len(x) > 1:
            return int(x.replace('K', '')) * 1000
        return 1000
    if 'M' in x:
        if len(x) > 1:
            return int(x.replace('M', '')) * 1000000
        return 1000000
    if 'B' in x:
        return int(x.replace('B', '')) * 1000000000
    return 1000000000


def days_from_human_format(x):
    if 'm' in x:
        return float(x.replace('m', '')) * (1/1440)
    if 'h' in x:
        return float(x.replace('h', '')) * (1/24)
    if 'd' in x:
        return float(x.replace('d', ''))
    if 'w' in x:
        return float(x.replace('w', '')) * 7
    if 'y' in x:
        return float(x.replace('y', '')) * 356
    return x

"""
    This method extract the number of followers, number of likes
    and number of the user's friends that liked the page the driver is in.
"""
def get_page_numbers(driver):
    likes = None
    mutuals = 0
    follows = None
    last_resort = 0

    # First checking if the numbers are in the home page of the page
    elements = driver.find_elements_by_xpath("//div[@class='lpgh02oy']/div/div")
    for elem in elements:
        if "About" in elem.text:
            about_arr = elem.text.split('\n')
            for det in about_arr:
                if 'people' in det:
                    if 'like' in det:
                        det_arr = det.split(' ')
                        likes = int(det_arr[0].replace(",", ""))
                        mutuals = int(det_arr[5].replace(",", ""))
                    if 'follow' in det:
                        det_arr = det.split(' ')
                        follows = int(det_arr[0].replace(",", ""))
            return follows, likes, mutuals

        if "Intro" in elem.text:
            intro_arr = elem.text.split('\n')
            for det in intro_arr:
                if 'Followers' in det:
                    det_arr = det.split(' ')
                    # taking the estimation of followers
                    last_resort = int_from_human_format(det_arr[0].replace(",", ""))

    # if not, checking in the 'about' page of the page
    redirect(driver, 'About')
    elements = driver.find_elements_by_xpath("//div[@class='je60u5p8']/div")
    for elem in elements:
        if 'people' in elem.text:
            if 'like' in elem.text:
                det_arr = elem.text.split(' ')
                likes = int(det_arr[0].replace(",", ""))
                mutuals = int(det_arr[5].replace(",", ""))
            if 'follow' in elem.text:
                det_arr = elem.text.split(' ')
                follows = int(det_arr[0].replace(",", ""))

    # if the exact number of followers is not given, then there is an estimation I called "last_resort",
    # which is taken above.
    if follows is None:
        follows = last_resort

    return follows, likes, mutuals


"""
    This method extract the age of the group the driver is in.
"""
def get_age_of_group(driver):
    fields = driver.find_elements_by_xpath(about_group_fields)
    for field in fields:
        if 'History' in field.text:
            hist_arr = field.text.split('\n')
            date_arr = hist_arr[1].replace(".", "").split(' ')
            date = ' '.join(date_arr[3:6])
            new_date = reformat_date(date)
            return calculate_age(new_date)
    return 0


"""
    this method extract the number of account that are in the group that driver is in.
"""
def get_friends_num_of_group(driver):
    fields = driver.find_elements_by_xpath(about_group_fields)
    for field in fields:
        if "total members" in field.text:
            friends_arr = field.text.split(' ')
            return int(friends_arr[0].replace(',', ''))


"""
    This method extract the number of the user's friends that are in the group the driver is in.
"""
def get_mutuals_group(driver):
    elementa = driver.find_elements_by_xpath("//span[@class='d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d3f4x2em fe6kdd0r mau55g9w c8b282yb iv3no6db jq4qci2q a3bd9o3v knj5qynh m9osqain']")
    for elem in elementa:
        if 'friends' in elem.text:
            arr_text = elem.text.split(' ')
            for i, word in enumerate(arr_text):
                try:
                    x = int(word.replace(',', ''))
                    return x+i-1
                except:
                    continue
    return 0


# making sure the given url doesn't ends with '/'
def check_url(url):
    if url[len(url)-1] == '/':
        return  url[0:len(url)-1]
    return url


"""
    This method gather all the information about an account
"""
def scrap_account(driver, account_url):

    # if the user is logged in, we simply bring the driver to the given account url
    # if not, we cannot be sure Facebook will be in English,
    # so we add suffix for the url so we get the English version of Facebook
    # account_url = check_url(account_url)
    if is_logged_in:
        driver.get(account_url)
    else:
        driver.get(account_url + to_english)
    time.sleep(2)

    # getting the name of the given account url
    try:
        name = driver.find_element_by_xpath("//div[@class='rq0escxv l9j0dhe7 du4w35lb j83agx80 cbu4d94t pfnyh3mw d2edcug0 hpfvmrgz p8fzw8mz pcp91wgn iuny7tx3 ipjc6fyt']")
        user_name = name.text
    except:
        return None

    # if the user is logged in, then they could be friends,
    # so we will want to find for how long they were friends and how many mutual friends they have.
    # if the user is not logged in, we consider those arguments to be 0.
    friendship_duration = 0
    mutual_friends = 0
    if is_logged_in:
        friendship_duration = extract_friendship_duration(driver)
        if friendship_duration is None:
            return None
        friendship_duration = int(friendship_duration)
        mutual_friends = extract_mutual_friends(driver, friendship_duration > 0)

    if is_logged_in:
        driver.get(account_url)
    else:
        driver.get(account_url + to_english)
    time.sleep(2)

    total_friends, age_of_account = scrap_about(driver)

    return Account(user_name, total_friends, age_of_account, friendship_duration, mutual_friends)


"""
    this method gather all the information of a given page.
"""
def scrap_page(driver, page_url):
    # if the user is logged in, we simply bring the driver to the given page url
    # if not, we cannot be sure Facebook will be in English,
    # so we add suffix for the url so we get the English version of Facebook
    # page_url = check_url(page_url)
    if is_logged_in:
        driver.get(page_url)
    else:
        driver.get(page_url + to_english)
    time.sleep(2)

    # initializing the variables
    mutual_friends = None
    page_age = 0
    followers = 0
    likes = 0

    # here we check, by trying to extract the name of the page, if we can scrap the page
    # if we can, is either because the user is logged in, or because the page allow unsucribeds to see it (such as theShadow)
    # there are to options of title for page - Hebrew page and English page
    try:
        # trying first option for name
        name = driver.find_element_by_xpath(
            "//div[@class='rq0escxv l9j0dhe7 du4w35lb j83agx80 cbu4d94t g5gj957u d2edcug0 hpfvmrgz on77hlbc buofh1pr o8rfisnq ph5uu5jm b3onmgus ihqw7lf3 ecm0bbzt']")
    except:
        name=""
    try:
        # if the first option didn't work we try a second
        if name == "":
            name = driver.find_element_by_xpath(
                "//div[@class='rq0escxv l9j0dhe7 du4w35lb j83agx80 cbu4d94t pfnyh3mw d2edcug0 hpfvmrgz p8fzw8mz pcp91wgn iuny7tx3 ipjc6fyt']")
        page_name = name.text.split('\n')[0]


        # if we did succeed to get a name (either from 1st or 2nd option) we gather all the other attributs.
        page_age = get_age_of_page(driver)
        followers, likes, mutual_friends = get_page_numbers(driver)

    # if neither of the options worked, then the user isn't logged in, and the page looks diffrent because of it.
    except:
        # we gather all the attributes of the page that we can (not mutual friends).
        try:
            name = driver.find_element_by_xpath("//span[@class='_kao']")
        except:
            return None
        page_name = name.text
        elements = driver.find_elements_by_xpath("//div[@class='_1xnd']/div")
        for elem in elements:
            arr = elem.text.split('\n')
            for cell in arr:
                if 'people like' in cell:
                    cell_arr = cell.split(' ')
                    likes = int(cell_arr[0].replace(',', ''))
                elif 'people follow' in cell:
                    cell_arr = cell.split(' ')
                    followers = int(cell_arr[0].replace(',', ''))
                elif 'Page created' in cell:
                    date = reformat_date(cell.replace('Page created - ', ''))
                    page_age = calculate_age(date, False)
                    break

    return Page(page_name, page_age, followers, likes, mutual_friends)



"""
    this method gathers all the information of a given group.
"""
def scrap_group(driver, group_url):

    # if the user is logged in, we simply bring the driver to the given group url + the "/about" suffix
    # because we want to get to the 'about' page of the group.
    # if the user isn't logged in, we cannot be sure Facebook will be in English,
    # so we add suffix for the url so we get the English version of Facebook + the "/about" suffix
    group_url = check_url(group_url)
    if is_logged_in:
        driver.get(group_url+'/about')
    else:
        driver.get(group_url + '/about'+to_english)
    time.sleep(2)

    # getting the other attributes of the group, that do not care of the user is logged in.
    name = driver.find_element_by_xpath(name_of_page)
    group_name = name.text.split('\n')[0]

    group_age = get_age_of_group(driver)
    friends_num = get_friends_num_of_group(driver)

    # getting mutual friend only if logged in, else returning None
    mutuals = None
    if is_logged_in:
        mutuals = get_mutuals_group(driver)
    return Group(group_name, group_age, friends_num, mutuals)


"""
    This method helps us scroll over posts of the the account the driver is in.
    elements_xpath - the xpath of posts
    num - number of posts to scrap
    arr - the array we want to enter the posts to
    this method only gets the actual text of the post and not the other parameters.
"""
def scroll_over_posts(driver, elements_xpath_text, elements_xpath_background, num, arr):
    # using the method of "send_keys" to type the END key to get to the end of the page.
    counter = 0
    index = len(arr)
    actions = ActionChains(driver)

    # elements = driver.find_elements_by_xpath(elements_xpath)
    text_elements = driver.find_elements_by_xpath(elements_xpath_text)
    background_elements = driver.find_elements_by_xpath(elements_xpath_background)
    old_elements_amount = len(text_elements) + len(background_elements)

    while counter < num:
        time.sleep(2)
        actions.send_keys(Keys.END)
        actions.perform()

        text_elements = driver.find_elements_by_xpath(elements_xpath_text)
        background_elements = driver.find_elements_by_xpath(elements_xpath_background)

        for post in text_elements:
            # if there is actual text in the post
            if post.text != "":
                # if the post is longer than usual, then there is the button of "see more"
                # we click it to get the full text of the post.
                if "See More" in post.text:
                    try:
                        more = redirect_by_xpath("//div[text()='See More']") #TODO: prefection it
                        # more[0].click()
                        # more = post.find_element_by_xpath("//div[text()='See More']")
                        if more is not None:
                            webdriver.ActionChains(driver).move_to_element(more[0]).click(more[0]).perform()
                    except:
                        pass
                # replacing all new-lines in post to spaces
                # and inserting the post to the given array if not already there
                text = post.text.replace('\n', ' ')
                if text not in arr and text != '':
                    arr.insert(index, text)
                    index += 1
                    counter += 1

        for post in background_elements:
            # if the post is longer than usual, then there is the button of "see more"
            # we click it to get the full text of the post.
            if "See More" in post.text:
                try:
                    more = redirect_by_xpath("//div[text()='See More']") #TODO: prefection it
                    # more[0].click()
                    # more = post.find_element_by_xpath("//div[text()='See More']")
                    if more is not None:
                        webdriver.ActionChains(driver).move_to_element(more[0]).click(more[0]).perform()
                except:
                    pass
            # replacing all new-lines in post to spaces
            # and inserting the post to the given array if not already there
            text = post.text.replace('\n', ' ')
            if text not in arr and text != '':
                arr.insert(index, text)
                index += 1
                counter += 1

        if len(text_elements) + len(background_elements) == old_elements_amount:
            break
        old_elements_amount = len(text_elements) + len(background_elements)
    return arr

"""
    this method extract from given account its posts by calling to scroll_over_posts function
    num - the number of posts to extract
"""
def scrap_posts(driver, account_url, num):
    driver.get(account_url)
    time.sleep(2)
    arr = []
    while len(arr) < num:

        arr = scroll_over_posts(driver, "//div[@class='kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql ii04i59q']", "//div[@class='sfj4j7ms pvbba8z0 rqr5e5pd dy7m38rt j7igg4fr']", num, arr)
        # arr = scroll_over_posts(driver, "//div[@class='sfj4j7ms pvbba8z0 rqr5e5pd dy7m38rt j7igg4fr']", num, arr,"Background")
        # scroll_over_posts(driver, "//span[@class='d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d3f4x2em fe6kdd0r mau55g9w c8b282yb iv3no6db jq4qci2q a3bd9o3v b1v8xokw oo9gr5id hzawbc8m']", num, arr)
        # scroll_over_posts(driver, "//div[@class='rq0escxv l9j0dhe7 du4w35lb hybvsw6c io0zqebd m5lcvass fbipl8qg nwvqtn77 k4urcfbm ni8dbmo4 stjgntxs sbcfpzgs']", "//span[@class='d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d3f4x2em fe6kdd0r mau55g9w c8b282yb iv3no6db jq4qci2q a3bd9o3v b1v8xokw oo9gr5id hzawbc8m']", num, arr)

    return arr

def scrap_comments(driver, num=0):
    comments = []
    post_comments = driver.find_elements_by_xpath("//div[@class='cwj9ozl2 tvmbv18p']/ul/li")
    arr_index = 0
    for i, comm in enumerate(post_comments):
        print(comm.text)
        lines = comm.text.split('\n')
        writer = lines[0]
        text = lines[1]
        likes = 0
        age = 0
        start_text = 1
        for j, line in enumerate(lines):
            if "Like" in line:
                try:
                    likes = int(lines[j-1])
                    text = ' '.join(lines[start_text:j-1])
                except:
                    likes = 0
                    text = ' '.join(lines[start_text:j])
            if "Share" in line or "Reply" in line:
                age_arr = line.split(' ')
                if is_logged_in:
                    age = age_arr[len(age_arr)-1]
                    if age == "Edited":
                        age = age_arr[len(age_arr)-2]
                    age = days_from_human_format(age)
                else:
                    try:
                        likes = int(lines[j - 1])
                        text = ' '.join(lines[start_text:j - 1])
                    except:
                        likes = 0
                        text = ' '.join(lines[start_text:j])
                    age = days_from_human_format(age_arr[2])
                if text == "":
                    text = None
                comments.insert(arr_index, {"Writer": writer, "Text": text, "Likes": likes, "Age": age})
                arr_index += 1
                if(len(lines) > j+1):
                    writer = lines[j+1]
                    start_text = j+2
        if arr_index >= num:
            break

    return comments

def click_on_all(driver, element_xpath):
    elements = driver.find_elements_by_xpath(element_xpath)
    print(len(elements))
    for elem in elements:
        # try:
        #     elem.click()
        # except:
        try:
            webdriver.ActionChains(driver).move_to_element(elem).click(elem).perform()
        except:
            print("couldn't click on element")
        time.sleep(1)
    return len(elements)


def scrap_one_post(driver, post_url, comments):
    post_url = check_url(post_url)
    if is_logged_in:
        driver.get(post_url)
    else:
        driver.get(post_url + to_english)

    # driver.get(post_url)
    time.sleep(2)

    sum = click_on_all(driver, "//span[@class='j83agx80 fv0vnmcu hpfvmrgz']")
    click_on_all(driver, "//div[text()='See More']")
    loops = 0
    while sum != 0 and loops < 10:
        sum = click_on_all(driver, "//span[@class='j83agx80 fv0vnmcu hpfvmrgz']")
        click_on_all(driver, "//div[text()='See More']")
        loops += 1

    post_writer = driver.find_elements_by_xpath("//h2[@class='gmql0nx0 l94mrbxd p1ri9a11 lzcic4wl aahdfvyu hzawbc8m']")
    if len(post_writer) == 1:
        writer = post_writer[0].text
    else:
        writer = None

    post_content = driver.find_elements_by_xpath("//span[@class='d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d3f4x2em fe6kdd0r mau55g9w c8b282yb iv3no6db jq4qci2q a3bd9o3v b1v8xokw oo9gr5id hzawbc8m']")
    if len(post_content) > 0:
        content = post_content[len(post_content)-1].text.replace("\n", " ")
    else:
        content = None

    if content == "" or content is None:
        post_content = driver.find_elements_by_xpath("//div[@class='sfj4j7ms pvbba8z0 rqr5e5pd dy7m38rt j7igg4fr']")
        if len(post_content) > 0:
            content = post_content[len(post_content) - 1].text.replace("\n", " ")
        else:
            content = None

    if content == "":
        content = None

    comments = scrap_comments(driver)


    # account = driver.find_elements_by_xpath("//span[@class='nc684nl6']")
    # accounts = driver.find_elements_by_xpath("//a[@class='oajrlxb2 g5ia77u1 qu0x051f esr5mh6w e9989ue4 r7d6kgcz rq0escxv nhd2j8a9 nc684nl6 p7hjln8o kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x jb3vyjys rz4wbd8a qt6c0cv9 a8nywdso i1ao9s8h esuyzwwr f1sip0of lzcic4wl oo9gr5id gpro0wi8 lrazzd5p']")
    # account = None
    # for elem in accounts:
    #     print(elem.text)
    #     if elem.text == writer:
    #         account = elem
    #         break
    # if account is not None:
    #     try:
    #         account.click()
    #     except:
    #         webdriver.ActionChains(driver).move_to_element(account).click(account).perform()
    #     # account[0].click()
    #     time.sleep(2)
    #     print(driver.current_url)
    actions = ActionChains(driver)
    actions.send_keys(Keys.HOME)
    # actions.send_keys(Keys.HOME)
    actions.perform()
    redirect(driver, writer, False)
    time.sleep(3)
    # not_wanted_url = driver.current_url
    # while "groups" in driver.current_url:
    writer_page_btn = driver.find_elements_by_xpath("//span[@class='nc684nl6']")
    # counter = 0
    # for btn in writer_page_btn:
    #     if writer in btn.text:
    #         counter += 1
    #         if counter > 1:
    #             webdriver.ActionChains(driver).move_to_element(btn).click(btn).perform()
    #             time.sleep(3)
    #             if "groups" not in driver.current_url:
    #                 break
            # #

        # print(str(i) + ": " + btn.text)
        # actions.send_keys(Keys.END)
        # actions.perform()
        # redirect(driver, writer, False)
        # time.sleep(3)
    url_account = driver.current_url
    print(url_account)
    url_arr = url_account.split('/')
    myUser = ""
    for i, cell in enumerate(url_arr):
        if "user" in cell:
            myUser = "/".join(url_arr[i+1:])
            break
    print(myUser)
    url_account = "https://www.facebook.com/" + myUser
    print(url_account)
    url_account = check_url(url_account)
    account = scrap_account(driver, url_account)
    print(account)
    return Post(writer, content, comments, account)


def scrap_url(driver, url, posts=0, loging_in=False):
    global is_logged_in
    is_logged_in = loging_in
    url = check_url(url)

    # scarpping a post
    if "posts" in url or "permalink" in url:
        one_post = scrap_one_post(driver, url, posts)
        if one_post is None:
            return None
        return one_post

    # scarpping a group
    elif "groups" in url:
        group = scrap_group(driver, url)
        if group is None:
            return None
        group_posts = scrap_posts(driver, url, posts)
        group.set_posts(group_posts)
        return group

    else:
        account = scrap_account(driver, url) #trying to scrap account
        # scarpping a page
        if account is None:
            print("PPPAAAAGGGGGEEEE")
            page = scrap_page(driver, url)
            if page is None:
                return None
            else:
                page_posts = scrap_posts(driver, url, posts)
                page.set_posts(page_posts)
                return page

        # scarpping an account
        else:
            account_posts = scrap_posts(driver, url, posts)
            account.set_posts(account_posts)
            return account





if __name__ == '__main__':

    driver = init_sel() #to init the driver
    if not login(driver, "ofrishani10@walla.com", "Ls5035"):
        login(driver, "ofrishani10@walla.com", "Is5035") #login in - return true on success, false otherwise.

    account = scrap_url(driver, "https://www.facebook.com/uri.mazor.52", posts=20, loging_in=True) #to scrap something; this case an account

    finish_sel(driver) #to finish with the driver

    analyzed = analyze_facebook(account) #to analyze the something; this case the account
    print(vars(analyzed))

    # account = scrap_url(driver, "https://www.facebook.com/Gilad.Agam", posts=20, loging_in=True)
    # group = scrap_url(driver, "https://www.facebook.com/groups/wakeupeople", posts=20, loging_in=True)
    # post = scrap_url(driver, "https://www.facebook.com/groups/336084457286212/permalink/648330709394917", posts=20, loging_in=True)
    # page = scrap_url(driver, "https://www.facebook.com/TheShadow69", posts=20, loging_in=True)
    # page = scrap_url(driver, "https://www.facebook.com/Conspiralla/", posts=20, loging_in=True)
    # scrap_facebook(url_account="https://www.facebook.com/Gilad.Agam", posts=20, loging_in=True, user_url="https://www.facebook.com/ofri.shani.31", user_mail="ofrishani10@walla.com", user_password="Is5035")
    # account = scrap_facebook(url_account="https://www.facebook.com/Gilad.Agam", posts=20, loging_in=True, user_mail="ofrishani10@walla.com", user_password="Is5035")
    # account = scrap_facebook(url_account="https://www.facebook.com/uri.mazor.52", posts=20, loging_in=True, user_mail="ofrishani10@walla.com", user_password="Is5035")
    # scrap_facebook(url_account="https://www.facebook.com/noam.fathi", posts=40, loging_in=True, user_url="https://www.facebook.com/ofri.shani.31", user_mail="ofrishani10@walla.com", user_password="Is5035")
    # page = scrap_facebook(url_page="https://www.facebook.com/TheShadow69", posts=20, loging_in=True, user_mail="ofrishani10@walla.com", user_password="Is5035")
    # posts = scrap_facebook(url_group="https://www.facebook.com/groups/336084457286212", posts=20, onlyPosts=True, loging_in=True, user_mail="ofrishani10@walla.com", user_password="Is5035")
    # post = scrap_facebook(url_post="https://www.facebook.com/groups/336084457286212/permalink/648330709394917", posts=20,  loging_in=True, user_mail="ofrishani10@walla.com", user_password="Is5035")
    # post = scrap_facebook(url_post="https://www.facebook.com/groups/813369869468028/posts/1062235314581481", loging_in=True, user_mail="ofrishani10@walla.com", user_password="Is5035")
    # group = scrap_facebook(url_group="https://www.facebook.com/groups/wakeupeople", posts=20, loging_in=True, user_mail="ofrishani10@walla.com", user_password="Is5035")
    # page = scrap_facebook(url_page="https://www.facebook.com/Conspiralla/", posts=40, loging_in=True, user_mail="ofrishani10@walla.com", user_password="Is5035")
    # print(posts)
    #
    # if analyzed is not None:
    #     print(vars(analyzed))
    # else:
    #     print("somhow its None")
    # print_dict(analyze_account(account))
    # print(analyze_page(page))

    # print(analyze_group(group))
    # page: "https://www.facebook.com/TheShadow69")
    # page: "https://www.facebook.com/hapshuta")
    # "https://www.facebook.com/groups/bathefer1")
    # post-link: "https://www.facebook.com/permalink.php?story_fbid=1510260152643112&id=100009774256825")
    # "https://www.facebook.com/Gilad.Agam"
