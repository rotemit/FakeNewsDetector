#pip install vaderSentiment
#pip install googletrans
#pip install nltk
#pip install tkinter
#pip install re
#pip install csv
import tkinter as tk
import nltk
import re
import csv
from tkinter.scrolledtext import ScrolledText
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from googletrans import Translator
from deep_translator import GoogleTranslator
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
# translator = Translator()
rawText = 'none'
EnglishText = ''
lableTranslation = ''
labelSentiment = ''
pos_word_list=[]
neu_word_list=[]
neg_word_list=[]

# Finds the subjects of the post according to predetermined categorical words
# def find_subject(HebrewText):
#     language = translator.detect(HebrewText)
#     if language.lang == 'iw' or language.lang == 'he':
#         language = 'Hebrew'
#     elif language == 'en': language = 'English'
#     else: language = language.lang
#
#     print(language)
#     return ''

def translateAndAnalize(HebrewText=None):
    master = tk.Tk()
    #tk.Label(master, text="Translation:" ).grid(row=5, sticky=tk.N)
    #tk.Label(master, text="Sentiment Analysis:").grid(row=6, sticky=tk.N)
    if HebrewText is None:
        HebrewText = st.get(1.0, 7000.0)
    EnglishText = GoogleTranslator(source='he', target='en').translate(HebrewText)
    translationStr = EnglishText
    subject = ''
    # subject = find_subject(HebrewText)
    # if('לפיד' in HebrewText):
    #     subject+='Lapid;'
    # if ('גנץ' in HebrewText):
    #     subject+= 'Ganz;'
    # if ('בנט' in HebrewText):
    #     subject+= 'Bennett;'
    # if ('ביבי' in HebrewText or 'רוה"מ' in HebrewText or 'ראש הממשלה' in HebrewText):
    #     subject+= 'Bibi;'
    # if(subject==''):
    #     subject='None'

    sentimentDict = sid.polarity_scores(EnglishText)
    sentimentStr = sentimentDict.__str__()
    translationStrList = translationStr.split(".")
    lableTranslation = ''
    wordList = re.sub("[^\w]", " ", EnglishText).split()
    for word in wordList:
        if (sid.polarity_scores(word)['compound']) >= 0.1:
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.1:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)
    pos_word_str = pos_word_list.__str__()
    neg_word_str = neg_word_list.__str__()
    neu_word_str = neu_word_list.__str__()
    countPos = len(pos_word_list)
    countNeg = len(neg_word_list)
    countNeu = len(neu_word_list)
    countTotal = countNeg+countPos+countNeu
    sentimentCalc = 0
    if(countPos == countNeg):
        sentimentCalc = 0
    elif(countPos > countNeg):
        sentimentCalc = countPos/(countPos+countNeg)
    else:
        sentimentCalc = countNeg/(countPos+countNeg) * (-1)
    for i in range (len(translationStrList)):
        lableTranslation+=translationStrList[i]+'.\n'

    with open('sentiment_analysis_myFB.csv', mode='a', newline='') as sentiment_csv:
        sentiment_writer = csv.writer(sentiment_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        sentiment_writer.writerow([ sentimentCalc, sentimentDict['compound'], countNeg, countPos, countTotal, subject])
    # tk.Label(master,
    #          text="Original text:\n " + st.get(1.0, 7000.0), relief="solid", font="Times 10", bd=0).grid(row=3, sticky=tk.S)
    tk.Label(master,
             text="Original text:\n " + HebrewText, relief="solid", font="Times 10", bd=0).grid(row=3,
                                                                                                         sticky=tk.S)
    tk.Label(master,
         text="Translation:\n "+lableTranslation, relief="solid", font="Times 10", bd=0).grid(row=4, sticky=tk.S)
    tk.Label(master,
             text="Subject:\n " + subject, relief="solid", font="Times 10", bd=0).grid(row=5, sticky=tk.S)
    tk.Label(master,
        text="\n Positive:"+ countPos.__str__()+"\n"+"Neutral:"+ countNeu.__str__()+"\n"+'Negative:'+ countNeg.__str__()+"\n", relief="solid", font="Times 10", bd=0).grid(row=6, sticky=tk.N)
    tk.Label(master,
         text="Sentiment Analysis from library: "+sentimentStr, relief="solid", font="Times 10", bd=1).grid(row=7, sticky=tk.N)
    tk.Label(master,
             text="Sentiment Analysis manual: " + sentimentCalc.__str__(), relief="solid", font="Times 10", bd=1).grid(row=8,sticky=tk.N)
    print(sentimentStr)
   # tokenized_sentence = nltk.word_tokenize(translationStr)
    pos_word_list.clear()
    neg_word_list.clear()
    neu_word_list.clear()

 #   print('Positive:', pos_word_list)
 #   print('Neutral:', neu_word_list)
 #   print('Negative:', neg_word_list)
# master = tk.Tk()
# #master.geometry("400x250")
#
# st = ScrolledText(master, height=5, width=30)
# #tk.Label(master, text="User:").grid(row=3, column=0, sticky=tk.W)
# #e1 = tk.Entry(master).grid(row=3, column=1, sticky=tk.W)
# tk.Label(master,text="Enter Hebrew text :").grid(row=0, sticky=tk.N)
# st.grid(row=1, sticky=tk.N)
# tk.Button(master, text='Translate and Analyze', command=translateAndAnalize).grid(row=2,sticky=tk.N, pady=4)
# #tk.Label(master,text="Translation:").grid(row=5, sticky=tk.N)
# #tk.Label(master,text="Sentiment Analysis:").grid(row=6, sticky=tk.N)
# tk.mainloop()



#while (rawText!='0'):
#    rawText=input('Enter Hebrew text (To Exit enter 0):')
#    EnglishText = translator.translate(rawText)
#    print('Translation:',EnglishText.text)
#    print('Sentiment Analysis:',sid.polarity_scores(EnglishText.text))
#print('Thanks for using Sentiment Analyzer')

if __name__ == "__main__":
    string_1 = "בוקר טוב ומחבק, מוזמנות להכיר את ערכת החיבוק להתחלות חדשות שלנו, שיצרתי בתחילת השנה שעברה מתוך המון מחשבה על כל מה שקורה עכשיו סביבנו ובתוכנו ומה יכול לעזור לנו לשגשג בתוך כל זה. היא התחילה כערכה לשנה החדשה ונשארה כערכה להתחלות חדשות לכל מי שרוצה חיבוק טוב כשהיא יוצאת לדרך יש בה מור ולבונה כדי להביא את אנרגיית האדמה והאוויר עם גחלים להקטרה, יש בה קטורות נעימות ומנחמות של סנדלווד, קונוסים של ורדים כדי להעלות את תדר האהבה והתקשורת הטובה ,מרווה לבנה כדי לשחרר את המיותר ומקל פאלו סנטו כדי לזמן את הרצוי. מינרל של שאבה כדי לעשות טקס אישי של שאיבת האנרגיה שאתן כבר לא זקוקות לה . יש בה את אבן הסדולייט שעוזרת לנו להתחבר לתמונה הגדולה , לתכנית האלוהית ולזכור שהכל לטובתנו ,ומסר אישי מהדרקונים הקסומים... שנה טובה ומחבקת לכולנו! גוד וויטצ חנות מכושפת בחרושת 1 פרדס חנה"
    string_2 = "בוא רגע דוקטור, אל תפחד... הכבשים רצים להתחסן כי אתה מדען לא דוקטור אתה, אתה חלק שניתן להחליף בפארמאפיה (פארמה+מאפיה) המשתמשת בתקשורת להריץ פאניקה באנשים ושקרים כבדים ללא ביסוס מדעי וכולנו רואים איך התקשורת השמאלנית בישראל תורמת מדמה להפצת שקר הקורונה. שימו לב לעבירה פלילית של עיתונאים/רופאים ומדענים המתייגים אתכם - מתנגדי החיסון - כאוייבי העם ומפיצי מחלות כנהוג בכנופיית הממשל באמצעות אמצעי התקשורת והרדיו. כאמור ממומנים היטב ומפעילי גרפיקאים מכל הרמות להציג עובדה שאינה קיימת כלל ונתונים שקריים רבים המערפלים את החלטותיכם אם להתחסן או לאו. מדינת ישראל ריססה חומרים רבים בשמיים בשנים האחרונות טרם עליית הקרינה החודרת כעת מהחלל שהפסיקה תנועת המטוסים ואלו שעדיין עולים לאוויר - מבחינת נתונים הם יושבים במטוסים ומקבלים קרינה כמו בצילום סי.טי אך למשך כל הטיסה כולל הטייסים וצוות הטיסה. השמש משתנה ועברייני העושר משנים חוקים שהצליחו לאזן את העולם, להביא אותו לסחר יציב וסין תמיד היוותה איום על כולם אז איך פתאום כל התקשורת בעולם מקשיבים לשיטות הסיניות קומוניסטיות ונועלים לצאצאיכם את עתידם בגטו תקשורתי מחריד כל נפש בריאה. תהו בריאים ללא חיסונים מעברייני השמאל הרדיקלי המממנים מדעני שקר להלחיץ את הבריות על הבריאות שלהן... משחק נחמד להאבלים, לא מלחיצים אוכלוסיות לחסן עצמן כשהנתונים שקריים ומגיעים ממאפיית הפארמה וכנופיות הממשל השמאל קיצוניות, מדווחים בידי קרייני הטייקונים המניעים את השקר/סגר/אנטי-סחר/אנטי-חופש תנועה או אף חופש ההחלטה שנלקחו מכם במקביל למספרים אסטרונומיים של נדבקים והמשך הלחץ שתסכימו לחסן עצמכם עם חומר מסוכן שיוצר במעבדה בווהאן, סין - ומקורו בנגיף האיידס* שממנו למדו לחקות נגיף שמתפרץ רק לאחר שנים בגוף המארח, קחו בחשבון שפייסבוק מטומטם ופושע וככל הנראה יחסום כל מתנגדי החיסונים או בעצם - מתנגדי החיים וההחלטה החופשית או האנושיים שבינינו שרואים קדימה וקוראים ומחפשים פוסטים שעלולים להציל את חייהם ולהימנע מכפיית החיסונים העולמית, שמטרתה לדלל** אוכלוסיות ולגרום למניעת ילודה. באהבה תמיד, עדיאל חברכם הנאמן"
    # translateAndAnalize(string_1)
    # translateAndAnalize(string_2)
