import streamlit as st
import pandas as pd
import numpy as np
import pickle

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


from PIL import Image

st.header('Fortune Medical Associates')
st.image('images/TopPicture1.jpg', use_column_width = 'always')    

condition_options = ['Birth Control', 'Depression', 'Pain', 'Anxiety', 'Acne', 'Bipolar', 'Insomnia', 'WeightLoss', 'Obesity', 'ADHD', 'Other']
condition = st.selectbox("Enter your health condition info:", condition_options)

# Load the model 
with open('FortuneMedical_Model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the TF-IDF vectorizer
with open('TfidfVectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

stop_words = stopwords.words('english') 
class StemPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, data, y = 0):
        return self

    def transform(self, data, y = 0):
        normalized_corpus = data.apply(self.stem_doc)
        return normalized_corpus
       
    def stem_doc(self, doc):
        stemmer = SnowballStemmer('english')
        lower_doc = [token.lower() for token in word_tokenize(doc) if token.isalpha()]
        filtered_doc = [token for token in lower_doc if token not in stop_words]
        stemmed_doc = [stemmer.stem(token) for token in filtered_doc]
        return " ".join(stemmed_doc)

form1 = st.form(key = 'inputs')
input_review = form1.text_input("Enter your detailed review after medication:")
form1_button = form1.form_submit_button('Submit')

stemm = StemPreprocessor()
input_trans = stemm.transform(pd.Series([input_review]))

# predicting processed input
predictions = loaded_model.predict(input_trans)

if predictions == 1:
    st.success("Result: Medication worked, positive review")
else:
    st.error("Result: Medication didn't work, negative review, consult with Doctor")


