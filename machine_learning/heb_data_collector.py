from modules.Group import Group
from Scraper.scrapper import scrap_facebook
import csv



def get_group_posts(url_group):
    posts = scrap_facebook(url_group=url_group, posts=20, loging_in=True, user_mail='rotem.mitrany@gmail.com', user_password='Kar2335li', only_posts=True)
    return posts

def get_account_posts(url_account):
    posts = scrap_facebook(url_account=url_account, posts=10, loging_in=True, user_mail='rotem.mitrany@gmail.com', user_password='Kar2335li', only_posts=True)
    return posts

if __name__ == '__main__':
    header = ['text']
    posts_talkAboutEverythingGroup = get_group_posts('https://www.facebook.com/groups/503350980581730')
    posts_vaccinesAndSideEffectsLive = get_group_posts('https://www.facebook.com/groups/421434465970824/')
    posts_drLiorUngar = get_account_posts('https://www.facebook.com/dr.lior.ungar')
    with open('heb_posts.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for post in posts_talkAboutEverythingGroup:
            writer.writerow([post])
        for post in posts_vaccinesAndSideEffectsLive:
            writer.writerow([post])
        for post in posts_drLiorUngar:
            writer.writerow([post])
    print("finishing")

