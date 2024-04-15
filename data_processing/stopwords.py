import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords



def remove_stopwords(listterms):
    st = LancasterStemmer()
    docs = []
    stop_words = set(stopwords.words('english'))
    for terms in listterms:
        tokens = []
        for m in terms:
            if m not in stop_words:
                tokens.append(st.stem(m.lower()))
            
        docs.append(tokens)
        
    return docs