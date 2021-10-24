import requests
import time
import json

'''
    To run Nemo on Rotem's computer:
    1. in one terminal, open yap server: 
        1.1 cd C:\Users\Rotem\yapproj\src\yap
        1.2 yap api
    2. in another terminal, open NEMO server:
        2.1 cd C:\Users\Rotem\PycharmProjects\Nemo\NEMO
        2.1 uvicorn api_main:app --port 8090
'''

if __name__ == '__main__':
    #rest api client
    text = "עשרות אנשים מגיעים מתאילנד לישראל.\nתופעה זו התבררה אתמול בוועדת העבודה והרווחה של הכנסת."
    tokenized = 'false'
    # Escape double quotes in JSON.
    text = text.replace(r'"', r'\"')
    url = 'http://localhost:8090/multi_to_single?multi_model_name=token-multi&verbose=0'
    data = {"sentences": text,"tokenized": tokenized}
    headers = {'content-type': 'application/json', 'accept': 'application/json'}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    # with open('yap_analysis.json', 'w', encoding='UTF8') as outfile:
    #     json.dump(r.json(), outfile, indent=4, ensure_ascii=False)
    time.sleep(3)
    # response = r.json()
    print(r.json())
    # r.json()[0]['ents']['token']['nemo_multi_align_token'][0]['text']
    # for i in r.json():
    #     first_ndx = r.json()[i]['ents']['token']['nemo_multi_align_token']
    #     for j in first_ndx:
    #         print(str(j) + 'st keyword: '+first_ndx[j]['text'])
