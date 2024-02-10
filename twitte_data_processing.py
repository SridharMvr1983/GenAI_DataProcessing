pip install pandas

import pandas as pd 

data=pd.read_csv("C:\AI-ML\genAI\Twitter Data Processing\sample.csv")
data["text"]
pip install sklearn
pip install scikit-learn

from sklearn.feature_extraction.text import CountVectorizer
BOW=CountVectorizer()
document_matrix=BOW.fit_transform(data["text"])
#VOCABULARY FOR bow
BOW.vocabulary_


from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Initialize OneHotEncoder
encoder = OneHotEncoder()


# Fit and transform the data
one_hot_encoded_data = encoder.fit_transform(data["text"])

one_hot_encoded_data
#Assignement2 : You need to perform BOW from the text column
bow_vector=document_matrix.toarray()
bow_vector


#Here you can try N-grams also like 2-gram, 3-gram,4-gram
bigram=CountVectorizer(ngram_range=(2,2))
bigram_vocab=bigram.fit_transform(data["text"])
bigram.vocabulary_
trigram=CountVectorizer(ngram_range=(3,3))
trigram_vocab=trigram.fit_transform(data["text"])
trigram.vocabulary_
fourgram=CountVectorizer(ngram_range=(4,4))
fourgram_vocab =fourgram.fit_transform(data["text"])
fourgram.vocabulary_
bigram_vocab.toarray()
trigram_vocab.toarray()
fourgram_vocab.toarray()


#At the end you have to perform the tf-idf 
from sklearn.feature_extraction.text import TfidfVectorizer
tifidf=TfidfVectorizer()
tifidf.fit_transform(data["text"]).toarray()
tifidf.vocabulary_
tifidf.idf_
