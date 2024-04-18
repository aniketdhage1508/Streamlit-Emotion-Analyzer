import streamlit as st
import time as ts
from datetime import time 
import numpy as np
import pandas as pd




def stream_data(text_data):
        for word in text_data.split(" "):
            yield word + " "
            ts.sleep(0.02)
        


def run():
    html_temp = """
    <div style="background-color:skyblue;padding:25px 10px;margin-bottom:50px;">
    <h1 style="color:white; text-align:center; font-size:50px; font-weight:bold;text-shadow: 2px 2px 4px #000000;">SENTIMENT ANALYZER</h1>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown(
    """
    <div style='border: 4px solid #447ECC; border-radius: 15px; margin-bottom: 10px;'>
        <h1 style='text-align: center; color: #E8D9CF;'>Deciphering Feelings through Words</h1>
    </div>
    """,
    unsafe_allow_html=True
)

    st.markdown("---")
    text_data = """
    **Problem Statement:**
    In many online platforms, understanding the emotional response of users to content is vital for improving user experience, engagement, and overall satisfaction. However, manually analyzing large volumes of text data to determine emotional states is time-consuming and impractical. Therefore, the need for automated emotion detection systems arises. These systems must accurately predict emotions from text data to provide valuable insights for various applications.

    **Introduction:**
    The project aims to develop an Emotion Prediction system using text data. Emotion prediction from text is crucial in various applications such as sentiment analysis, customer feedback analysis, mental health monitoring, and chatbot interactions. By analyzing text data and predicting emotions, this system can provide valuable insights into the emotional states of individuals, sentiment trends, and user interactions.

    **Solution:**
    The solution involves building a machine learning model that can predict emotions from text data. By leveraging Natural Language Processing (NLP) techniques, the system can analyze textual content and extract features that represent emotional states. These features are then used to train machine learning models to predict emotions accurately. The system undergoes several stages:

    1. **Exploratory Data Analysis (EDA):** Analyzing the dataset to understand the distribution of emotions, the frequency of different emotional categories, and any patterns or trends present in the data.

    2. **Preprocessing:** Cleaning and preprocessing the text data by removing noise, tokenizing text into words or sentences, removing stop words, and performing other text normalization techniques to prepare the data for model training.

    3. **Model Building:** Developing machine learning models using various algorithms such as LSTM (Long Short-Term Memory), Naive Bayes, Random Forest, Gradient Boosting, and SVM (Support Vector Machines). These models are trained on the preprocessed text data to learn the relationship between textual features and emotional categories.

    4. **Evaluation:** Evaluating the performance of each model using metrics such as accuracy, precision, recall, and F1 score. This helps in assessing how well each model generalizes to unseen data and how effectively it predicts emotions from text.

    5. **Comparison:** Comparing the performance of all the models based on the evaluation metrics to identify the most effective model for emotion prediction. This comparison helps in selecting the best model for deployment in real-world applications.

    By implementing this solution, we can build an automated Emotion Prediction system that effectively analyzes text data and predicts emotional states, providing valuable insights for various applications.
    """
    st.write_stream(stream_data(text_data))
    
    # Add footer
    st.markdown(
        """
        <style>
        .footer {
            padding: 10px;
            color: white;
            background-color: #333;
            text-align: left;
            font-size:20px
        }
        </style>
        """,
        unsafe_allow_html=True
    )



    # Define the URLs for the social media profiles
    social_media_urls = {
        "Instagram": "https://www.instagram.com/your_instagram_username/",
        "Twitter": "https://twitter.com/your_twitter_username/",
        "LinkedIn": "https://www.linkedin.com/in/your_linkedin_username/"
    }

    # Define the URLs for the social media logos
    social_media_logos = {
        "Instagram": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Instagram_icon.png/768px-Instagram_icon.png",
        "Twitter": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Twitter_logo.png/600px-Twitter_logo.png",
        "LinkedIn": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/600px-LinkedIn_logo_initials.png"
    }

    # Create the footer
    st.markdown("---")
    st.markdown("""<h4>Developed By : </h4>""", unsafe_allow_html=True)
    # Display the content inside a box
    st.markdown("""
        <div style='background-color: black; padding: 10px 15px;'>
            <ul>
                <li>Aniket Dhage</li>
                <li>Sanika Butle</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

