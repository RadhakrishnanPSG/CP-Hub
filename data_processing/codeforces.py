import requests
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from stopwords import remove_stopwords

def getCodeforces():
    cf = requests.get('https://codeforces.com/api/problemset.problems?tags=implementation')
    cf_json = cf.json()
    documents = []
    links = []
    cf_afterstop = []
    for i in cf_json['result']['problems']:
        tags = i['tags']
        name = i['name'].split()
        terms = tags+name+["codeforces"]
        cf_afterstop.append(remove_stopwords(terms))
        documents.append(tags+name)
        links.append(["https://codeforces.com/contest/"+str(i['contestId'])+"/problem/"+i['index']])        

    documents = remove_stopwords(documents)
    return documents, links


