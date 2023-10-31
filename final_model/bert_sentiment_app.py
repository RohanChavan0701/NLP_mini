import streamlit as st

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("model_bert")
tokenizer = BertTokenizer.from_pretrained("model_bert")


model.eval()


def predict_sentiment(review_text):

    inputs = tokenizer(review_text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():

        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item()


st.title("BERT Sentiment Analysis For Movie Reviews")

review = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if review:
        
        prediction = predict_sentiment(review)
        sentiment_labels = {0: "Negative", 1: "Positive"}
        st.write(f"Sentiment: {sentiment_labels[prediction]}")
    else:
        st.write("Please enter a review before predicting sentiment.")
