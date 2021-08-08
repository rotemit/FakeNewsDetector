import csv

from facebook_scraper import get_posts


def scrap_posts(type, url, write, num=10):
    counter = 0
    posts = None
    if type == "page":
        posts = get_posts(url)
    elif type == "group":
        posts = get_posts(group=url)
    elif type == "account":
        posts = get_posts(account=url)
    for post in posts:
        if post['post_text'] is not None and post['post_text'] != '':
            writer.writerow([post['post_id'], post['post_text'].replace("\n", ' '), post['time'], post['username']])
            counter += 1
        if counter == num:
            break


# https://pypi.org/project/facebook-scraper/#description
# if __name__ == "__main__":
#     file = open('file.csv', 'w', encoding='UTF8')
#     writer = csv.writer(file)
#     writer.writerow(['post_id', 'post_text', 'date', 'writer', 'label'])
#     scrap_posts("account", "Netanyahu", writer, 20)
#     scrap_posts("account", "NaftaliBennett", writer)
#     scrap_posts("account", "miri.regev.il", writer)
#     scrap_posts("account", "MichaeliMerav", writer)
#     scrap_posts("page", "Meretz", writer)
#     scrap_posts("account", "BeGantz", writer)
#     scrap_posts("account", "YairLapid", writer, 20)
#     file.close()
#     print("done")
