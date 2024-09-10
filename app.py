import re
import pickle  
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


st.title("Sentiment Analysis")
st.write("This is a sentiment analysis model for predicting nature of tweets.")



message = st.text_input("Enter tweet here: ", )

lem = WordNetLemmatizer()
corpus = []

for i in range(0, 1):
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower()
    review = review.split()
    review = [lem.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

with open('./models/Model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('./models/cv.pkl', 'rb') as file:
    cv = pickle.load(file)
# Note: Accuracy of this model is 77.06


if st.button("Predict"):
    if (model.predict(cv.transform(corpus).toarray())) == 2:
        st.write("Tweet is Positive...")
    elif ((model.predict(cv.transform(corpus).toarray())) == 1):
        st.write("Tweet is Neutral...")
    else:
        st.write("Tweet is Negative...")
