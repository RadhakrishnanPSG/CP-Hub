import ast
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask, render_template, request
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__, template_folder= 'template')

with open('C:\ir\Coding-contest-search-engine\data_processing\documents.txt', 'r', encoding='utf-8') as f:
    for i in f:
        documents = ast.literal_eval(i)
        break

with open('C:\ir\Coding-contest-search-engine\data_processing\links.txt', 'r', encoding='utf-8') as f:
    for i in f:
        links = ast.literal_eval(i)
        break


for i in range(len(documents)):
        documents[i] = " ".join(documents[i])
    
vectorizer = TfidfVectorizer() 
vectors = vectorizer.fit_transform(documents)

tf_idf = pd.DataFrame(vectors.todense()).iloc[:-1]  
tf_idf.columns = vectorizer.get_feature_names_out()
tfidf_matrix = tf_idf.T
tfidf_matrix['count'] = tfidf_matrix.sum(axis=1)
tfidf_matrix = tfidf_matrix.sort_values(by ='count')

names = list(tfidf_matrix['count'].index.values)

def getresults(query):
    finallinks = []
    st = LancasterStemmer()
    c_query = [st.stem(i.lower()) for i in query.split(" ")]
    cs = [0]*len(names) 

    for i in range(len(names)):
        for j in range(len(c_query)):
            if names[i] == c_query[j]:
                cs[i]+=1

    cs_df = pd.DataFrame(cs)

    rank = []
    for i in range(tfidf_matrix.shape[1]-1):
        rank.append((i, cosine_similarity(cs_df.T, pd.DataFrame(tfidf_matrix[i]).T)[0]))

    
    ind = sorted(rank, key = lambda x:x[1], reverse = True)[:20]
    if ind[0][1]==0:
        # print("Not found")
        finallinks = ["Not found"]
        pass
    else:
        for i,j in ind:
            if j!=0:
                finallinks.append(links[i][0])

    return finallinks

def func(text):
  # Remove all special characters using regular expressions
  cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
  cleaned_text=cleaned_text.lower()
  return cleaned_text

def getresults_transformers(query):
    df = pd.read_csv("C:\ir\Coding-contest-search-engine\data_processing\leetcode_questions.csv")
    df['Question Text'] = df['Question Text'].fillna('')
    df['Question Text']=df['Question Text'].apply(func)
    question1=df['Question Text'].tolist()
    questions_title = df['Question Slug'].tolist()
    d=dict()
    for i,j in zip(question1,questions_title):
        d[i]=j
    questions = [item for item in question1 if item != ""]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding=[]
    for question in questions:
        embedding.append(model.encode(question, convert_to_tensor=True))
    # query = "you are given a 0indexed 2d array grid of size 2 x n "
    query=func(query)
    query_emb=model.encode(query, convert_to_tensor=True)
    score=[]
    for emb in embedding:
        score.append(util.pytorch_cos_sim(emb, query_emb).item())
    # Get the indices of questions sorted by similarity in descending order
    similar_question_indices = np.argsort(score)[::-1]

    # Number of top similar questions to retrieve
    num_similar_questions = 10

    # Retrieve the top similar questions
    top_similar_questions = ["https://leetcode.com/problems/"+d[questions[i]]+ "/description/" for i in similar_question_indices[1:num_similar_questions + 1]]
    return top_similar_questions
    
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method=="GET":
        result = []
        return render_template("index.html", result = result, len = len(result))
    else:
        query = request.form.get('search')
        result = getresults_transformers(query)
        return render_template("index.html", result = result, len = len(result))

if __name__=='__main__':
    app.run()



