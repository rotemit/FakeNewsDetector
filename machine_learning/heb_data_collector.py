from modules.Group import Group
from Scraper.scrapper import scrap_facebook
import csv

'''
    posts we took from facebook for testing
'''
covid_posts = [
    ['"לא ידענו שזה השלום האחרון": יותר מ-550 חולי קורונה נפטרו החודש בישראל. בני המשפחה, החברים וגם בעלי החיים לא מפסיקים להתגעגע'], #kan news
    ['מבצע החיסונים מוצלח מאוד, הן בקצבו והן בתוצאותיו. לאור זאת, אנחנו מרחיבים את האפשרות לקבל חיסון שלישי לכלל האוכלוסייה, ובתנאי שחלפו 5 חודשים מאז המנה השנייה. צאו להתחסן, בין אם זה חיסון ראשון, שני או שלישי - זה חשוב לכל אחד מכם וזה חשוב לכולנו כאוכלוסייה!'], #health ministry
    ['תראו איך מרעילים את הילדים באמצעות הבדיקה ששלחו לנו מבתי הספר והגנים!!!!🙄 לא לעשות!!!!'], #נפגעי חיסון הקורונה
    ['פייזר. ארגון פשע חוקי. שוחד, הסתרת נזקים, ייצור והפצה של סמים פסיכיאטריים ושאר רעלים במסווה של ״רפואה״ וקנסות של מאות מיליוני דולרים (רק על מה שהם נתפסו).'], #mor sagmon page
    ['אז לגבי המרכיבים שבזריקות, מסתבר שישנם רכיבים שלא דווחו'],  #vaccine choice il
    ['פנינים שנאמרו בזום המורים. ושימו לב, 25% מבעלי התו הירוק קיבלו פלסיבו'],  #vaccine choice il
    ['קונספירציה חדשה: הבדיקות המהירות שחילקו בבתי הספר הם לא ל covid 19. עבדו עלינו, זה לסארס 2!!!!! אנשים הזויים. יכול להבין שאנשים לא יודעים סתם ככה, אבל מה קשה לעשות גוגל?!'],  #מה כבר יכול לקרות
    ['קורונה מסוכנת וגורמת למוות'],  #rotem
    ['Covid-19 גורמת למוות'],  #rotem
    ['ארגון הבריאות העולמי מכנה רשמית את הקורונה בתור קוביד-19'],  #
    ['תמונה מראה אנשים שנדבקו בקורונה שוכבים על המדרכה בסין'],  #
    ['כשבועיים מאז החל מבצע החיסונים בחיסון השלישי, חצתה היום מדינת ישראל את רף מיליון המתחסנים בחיסון השלישי. מדובר ביותר ממחצית האוכלוסייה ברת החיסון (כ-1.9 מיליון בני 50+, שחלפו חמישה חודשים מאז קיבלו את מנת החיסון השנייה).'],  #
    ['ההבדלים בין החיסון של מודרנה לבין זה של פייזר קטנים מאד מכל הבחינות. אותה טכנולוגיה ויעילות ובטיחות דומים. לגבי דלקת בשריר הלב - רק אצל הרופא והקרדיולוג.'], #מדברימדע בתגובות
    ["מחקר אמריקאי חושף: תינוקות וילדי 'דור הקורונה' נפגעים בגלל הסגרים, הבידוד וגם בשל מצוקות הוריהם, מתקשים בדיבור, בהבנה ובביצוע פעולות מוטוריות. ('ידיעות')"],  #מדברים קורונה
    # [''],  #
    # [''],  #
    # [''],  #
]

def get_group_posts(url_group):
    posts = scrap_facebook(url_group=url_group, posts=20, loging_in=True, user_mail='rotem.mitrany@gmail.com', user_password='Kar2335li', onlyPosts=True)
    return posts

def get_account_posts(url_account):
    posts = scrap_facebook(url_account=url_account, posts=10, loging_in=True, user_mail='rotem.mitrany@gmail.com', user_password='Kar2335li', onlyPosts=True)
    return posts

if __name__ == '__main__':
    header = ['text']
    posts_talkAboutEverythingGroup = get_group_posts('https://www.facebook.com/groups/503350980581730')
    posts_vaccinesAndSideEffectsLive = get_group_posts('https://www.facebook.com/groups/421434465970824/')
    posts_drLiorUngar = get_account_posts('https://www.facebook.com/dr.lior.ungar')
    with open('heb_posts.csv', 'a', encoding='UTF8', newline='') as f: #open for appending
        writer = csv.writer(f)
        writer.writerow(header)
        for post in posts_talkAboutEverythingGroup:
            writer.writerow([post])
        for post in posts_vaccinesAndSideEffectsLive:
            writer.writerow([post])
        for post in posts_drLiorUngar:
            writer.writerow([post])
    print("finishing")

