import pandas as pd
import re
import nltk
import streamlit as st
import pickle
import json
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def download_nltk_data():
    try:
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
download_nltk_data()

st.title("Airline Sentiment Analysis")
st.write("Enter A Movie Review To Classify It As Positive Or Negative")
# get input
statement = st.text_area("Enter Your Review")

# Cleaning The Data
def get_words (data) :
    text = data
    url_pattern = r'https?://\S+|www\.\S+|t\.co/\S+'
    email_pattern = r'\b\w+@\w+\.\w+\b'
    username_pattern = r'@\w+'
    emoji_pattern = r'[\U00010000-\U0010FFFF]'

    text_cleaned = re.sub(url_pattern, '', text)
    text_cleaned = re.sub(email_pattern, '', text_cleaned)
    text_cleaned = re.sub(username_pattern, '', text_cleaned)
    text_cleaned = re.sub(emoji_pattern, '', text_cleaned)

    words = re.findall(r'\b[a-zA-Z]+\b', text_cleaned)
    return words

# importing the models
with open('model.pkl', 'rb') as file :
    model = pickle.load(file)
with open('countVectorizer.pkl', 'rb') as file:
    count_vec = pickle.load(file)
with open('stopWords.json', 'r') as file :    
    stop = json.load(file)

def get_simple_pos (tag) :
    if tag.startswith('J') :
        return wordnet.ADJ
    elif tag.startswith('V') :
        return wordnet.VERB
    elif tag.startswith('N') :
        return wordnet.NOUN
    elif tag.startswith('R') :
        return wordnet.ADV
    else :
        return wordnet.NOUN

def clean_document (words) :
    # it should not be a stop word and we have to lemmatize it by getting pos tag
    output_words = []
    for w in words :
        if w.lower() not in stop :
            pos = pos_tag([w])
            clean_word = lemmatizer.lemmatize(w, pos = get_simple_pos(pos[0][1]))
            output_words.append(clean_word)
    return output_words

if st.button('Classify') :
    words = get_words(statement)
    input = clean_document(words)
    input_str = ' '.join(input) 
    input_new = count_vec.transform([input_str]).toarray()
    prediction = model.predict(input_new)[0]
    st.write(f'Sentiment : {prediction}')
else :
    st.write("Please Enter A Movie Review")
    
    
    