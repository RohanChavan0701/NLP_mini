import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


with open('review_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


with open('tfidf_data.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer, tfidf_matrix = pickle.load(tfidf_file)


st.title('Sentiment Analysis App')


user_input = st.text_area('Enter your review:')


def predict_sentiment(sample_review):
    sample_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split()
    sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in sample_review_words]
    final_review = ' '.join(final_review)

    temp = tfidf_vectorizer.transform([final_review])


    prediction = model.predict(temp)
    return prediction

if st.button('Predict Sentiment'):
    if user_input:
       
        prediction = predict_sentiment(user_input)

        
        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        if prediction[0] in sentiment_labels:
            st.write(f'Sentiment: {sentiment_labels[prediction[0]]}')
        else:
            st.write('Unknown sentiment (Value: {})'.format(prediction[0]))
    else:
        st.write('Please enter a review before predicting sentiment.')
