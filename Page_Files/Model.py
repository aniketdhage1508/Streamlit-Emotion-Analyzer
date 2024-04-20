import streamlit as st
import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences   # Padding sequences for equal length input to neural networks
import re
import tensorflow as tf
import contractions
import nltk
# Download NLTK resources
nltk.download('stopwords')   # Download stopwords resource
nltk.download('punkt')   # Download punkt resource for tokenization
nltk.download('wordnet')   # Download wordnet resource for lemmatization
from nltk.corpus import stopwords
from bs4 import BeautifulSoup   # BeautifulSoup for HTML parsing
from nltk.tokenize import word_tokenize   # Tokenization for splitting text into words
from nltk.stem import WordNetLemmatizer, PorterStemmer   # Lemmatization and stemming for word normalization
import contractions   # Contractions for expanding contractions in text


def loading_the_files():
    with open('Files/lstm_model4.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('Files/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('Files/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    return loaded_model, label_encoder, tokenizer

##**Data Cleaning**

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_emojis(text):
    # Define the regular expression pattern to match emoji characters
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)

    # Replace emoji characters with an empty string
    return emoji_pattern.sub(r'', text)

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

def lowercase_text(text):
    return text.lower()

def Clean_Text(text):
    # Remove special characters, punctuation, numbers, and URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Removing Extra Spaces with Preceeding or successing Spaces
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def tokenize_text(text):
    # Basically splits the long sentences into tokens of word
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    # Sets the stop words in English like 'is', 'the' which have no importance
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()  # Creating the instance of WordNetLemmatizer Class
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatizing the words in the tokens
    return lemmatized_tokens

def stem_tokens(tokens):
    porter_stemmer = PorterStemmer()  # Creating the instance of PorterStemmer Class
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]  # Stemming the words in the tokens
    return stemmed_tokens

def Preprocessor(Tweet):
    Tweet = remove_html_tags(Tweet)
    Tweet = remove_emojis(Tweet)
    Tweet = expand_contractions(Tweet)
    Tweet = lowercase_text(Tweet)
    Clean_Tweets = Clean_Text(Tweet)
    Clean_Tweets = re.sub(r'[^\w\s]', '', Clean_Tweets)  # Remove punctuation marks
    Tokens = tokenize_text(Clean_Tweets)
    Tokens = remove_stopwords(Tokens)
    Lemmatized_Tokens = lemmatize_tokens(Tokens)
    Stemmed_Tokens = stem_tokens(Lemmatized_Tokens)

    return Stemmed_Tokens


def predict_emotion(text, loaded_model, label_encoder, tokenizer):
    
    processed_text = Preprocessor(text)
    tokenized_text = tokenizer.texts_to_sequences([processed_text])
    padded_text = pad_sequences(tokenized_text, maxlen=35, padding='post')
    # Make prediction
    emotion_probabilities = loaded_model.predict(padded_text)[0]
    # Get the predicted emotion class
    predicted_emotion_class = np.argmax(emotion_probabilities)
    # Map predicted class to emotion label
    predicted_emotion_label = label_encoder.classes_[predicted_emotion_class]
    
    return predicted_emotion_label

def btn_click(text_input, loaded_model, label_encoder, tokenizer):
    
    if text_input=="":
        return
    predicted_emotion_label = predict_emotion(text_input, loaded_model, label_encoder, tokenizer)
    return predicted_emotion_label




def run():
    html_temp = """
    <div style="background-color:skyblue;padding:25px 10px;margin-bottom:50px;">
    <h1 style="color:white; text-align:center; font-size:50px; font-weight:bold;text-shadow: 2px 2px 4px #000000;">SENTIMENT ANALYZER</h1>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader('This is a simple sentiment prediction app')
    # Load the model and label encoder
    loaded_model, label_encoder, tokenizer = loading_the_files()

        
    # Get user input
    text_input = st.text_area('Enter your text', max_chars=100, placeholder="Enter Some Text to Predict the Emotion Eg. I hate you")
        
    if st.button("Predict"):
        predicted_emotion_label = btn_click(text_input, loaded_model, label_encoder, tokenizer)
        if predicted_emotion_label==None:
            st.error("Enter the Text First")
        else:
            if len(text_input)<20:
                st.warning("This Prediction might be Inaccurate")
                st.success("The Emotion Predicted is : **{}**".format(predicted_emotion_label.capitalize()))
            else:
                st.success("The Emotion Predicted is : **{}**".format(predicted_emotion_label.capitalize()))
    # if col2.button("Text File"):
    #     st. title("Upload your File")
    #     st. markdown ("---")
    #     file = st.file_uploader("Please upload your File here",type=["csv"],accept_multiple_files=True)
    #     if file:
    #         st.write("Hello World")
    #         df = pd.read_csv(file)
    #         X = df['Text'].apply(Preprocessor)

    #         # Tokenize and pad the text data
    #         sequences = tokenizer.texts_to_sequences(X)
    #         X = pad_sequences(sequences, maxlen=35)
    #         predicted_emotions = loaded_model.predict(X)

    #         predicted_emotions = np.argmax(predicted_emotions)
    #         # Map predicted class to emotion label
    #         predicted_labels = label_encoder.classes_[predicted_emotions]
        
    #         df['Emotion'] = predicted_labels
    #         if st.button("Label"):
    #             st.write(df)
    #         if st.button("Download CSV"):
    #             csv = df.to_csv(index=False)
    #             b64 = base64.b64encode(csv.encode()).decode()  
    #             href = f'<a href="data:file/csv;base64,{b64}" download="predicted_emotions.csv">Download CSV File</a>'
    #             st.markdown(href, unsafe_allow_html=True)
