import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS spam classifier')

def transform_text(text):
    text = text.lower()   # to get all the alphabets in lower case
    text = nltk.word_tokenize(text)   # seperating all the words using space as separater
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)     # if word is alphabet or numeric then make new list of those words
    text = y[:]
    y.clear()   # copy of y
    for i in text:    #here text is list of tokens
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)  # again creating new list of elements which are not in stop words nor they are part of punctuation
    text = y[:]   # creating copy of y
    y.clear()   # clearing value of y
    for i in text:
        y.append(ps.stem(i))  # again append each word in y with applied stem function
    
    return " ".join(y)   # join all the elements in y using join function
# preprocessing of text


input_sms=st.text_input('Enter the message')

if st.button('Predict'):

    transformed_msg=transform_text(input_sms)

    # vectorization of text data

    vector_input=tfidf.transform([transformed_msg])

    #prediction

    result=model.predict(vector_input)[0]

    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')