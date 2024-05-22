import streamlit as st
from PIL import Image

# Add background image with absolute URL
background_image_url = "Visualization\Fig1.png"
st.markdown(
    f"""
    <style>
        body {{
            background-image: url('{background_image_url}');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Define image paths
image_paths = [
    "Visualization/Fig1.png",
    "Visualization/Fig2.png",
    "Visualization/Fig3.png",
    "Visualization/Fig4.png",
    "Visualization/Fig5.png",
    "Visualization/Fig6.png",
    "Visualization/Fig7.png",
    "Visualization/Fig8.png",
    "Visualization/Fig9.png",
    "Visualization/Fig10.png",
    "Visualization/Fig11.png",
    "Visualization/Fig12.png",
    "Visualization/Fig13.png",
    "Visualization/Fig14.png",    
]

# Display title and images
def run():
    # Title
    st.markdown(
    """
    <div style='border: 4px solid #447ECC; border-radius: 15px; padding: 20px; margin-bottom: 30px; background-color:#1B1F31 ;'>
        <h1 style='text-align: center; color: #E8D9CF;'>Graph Showcase</h1>
    </div>
    """,
    unsafe_allow_html=True
    )

    # Display images
    col1, col2, col3 = st.columns(3)
    for i, image_path in enumerate(image_paths):
        col_index = i % 3
        with [col1, col2, col3][col_index]:
            image = Image.open(image_path)
            st.image(image, use_column_width=True, caption=f"Fig {i+1}")
            st.markdown("<style>img {border-radius: 10px;}</style>", unsafe_allow_html=True)

    # Description box
    visualization_text="""

"""
    st.markdown(
        """
        <div style='border: 4px solid #447ECC; border-radius: 15px; padding: 20px; margin-top: 30px; background-color:#1B1F31 ;'>
            <h2 style='text-align: left; color: #darkblue;'>Description Box</h2>
            <ul style='color: white; padding-left: 20px;'>
                <li><b>Fig1<br/>Top 20 Most Repeating Words in Text Column :</b> This bar plot displays the top 20 most frequent words found in the text column of the dataset. Each bar represents a word, and its height indicates the frequency of occurrence in the text data. The colors represent different word categories, enhancing visual appeal.</li>
                <li><b>Fig2<br/>Distribution of Character Counts :</b> This KDE (Kernel Density Estimate) plot illustrates the distribution of character counts across all text samples in the dataset. The smooth curve indicates the density of character counts, with peaks indicating higher density regions. It helps visualize the variability in text length.</li>
                <li><b>Fig3<br/>Scatter Plot of Word Frequency :</b> This scatter plot visualizes the frequency of each word in the dataset against its rank. Words are sorted by frequency in descending order, and the x-axis is in logarithmic scale for better visualization. It helps identify the distribution of word frequencies and potential patterns.</li>
                <li><b>Fig4<br/>KDE Plots for Character Counts by Emotion :</b> These KDE plots show the distribution of character counts for each emotion category separately. Each color represents a different emotion, and the filled area under the curve depicts the density of character counts. It helps compare the character count distributions across different emotions.</li>
                <li><b>Fig5<br/>KDE Plots for Word Counts by Emotion :</b> Similar to Fig4, these KDE plots visualize the distribution of word counts for each emotion category. Each plot represents a different emotion, and the filled area under the curve indicates the density of word counts. It assists in comparing word count distributions across emotions.</li>
                <li><b>Fig6<br/>KDE Plots for Stopword Counts by Emotion :</b> These KDE plots demonstrate the distribution of stopword counts for each emotion category. Stopwords are common words like "the," "is," "and," etc., which often carry little meaning. Each plot represents a different emotion, and the filled area under the curve indicates the density of stopword counts.</li>
                <li><b>Fig7<br/>Top n-grams : </b>The code creates bar plots for the most frequent unigrams, bigrams, and trigrams in a text dataset. Each plot shows the top 30 n-grams, with bar heights representing their frequencies. The visualizations provide a clear overview of common words and phrases in the text.</li>
                <li><b>Fig8<br/>Unigram : </b>This bar plot displays the top 30 most frequent unigrams in the text data. Each bar represents a unigram, and its height indicates the frequency of occurrence. It helps to identify the most common individual words in the dataset.</li>
                <li><b>Fig9<br/>Bigram : </b>This bar plot shows the top 30 most frequent bigrams in the text data. It helps to identify common word pairs, providing insights into frequently occurring word sequences.</li>
                <li><b>Fig10<br/>Trigram : </b>This bar plot illustrates the top 30 most frequent trigrams in the text data. Each bar represents a trigram, and its height indicates the frequency of occurrence. This plot helps to identify common three-word sequences in the dataset.</li>
                <li><b>Fig11<br/>Distribution of Emotions :</b> This count plot displays the distribution of different emotions in the dataset. Each bar represents an emotion category, and its height indicates the frequency of occurrence. It provides an overview of the relative proportions of different emotions in the dataset.</li>
                <li><b>Fig12<br/>Word Cloud of Texts :</b> This word cloud visualizes the most frequent words in the text data. The size of each word corresponds to its frequency, with larger words appearing more frequently. It offers a visually appealing representation of the most common words in the dataset.</li>
                <li><b>Fig13<br/>Word Clouds for Each Emotion :</b> These word clouds depict the most frequent words associated with each emotion category separately. Each word cloud represents a different emotion, highlighting the words that are most characteristic of that emotion. It provides insights into the language associated with each emotion.</li>
                <li><b>Fig14<br/>Histogram of Text Length vs Frequency (with Emotion) :</b> This histogram illustrates the distribution of text lengths (character counts) across different emotions. Each color represents a different emotion category, and the height of each bar indicates the frequency of text samples with a particular length. It helps understand how text length varies across emotions.</li>
                <li><b>Fig15<br/>Distribution of Emotions :</b> This pie chart visualizes the distribution of different emotions in the dataset as percentages. Each slice represents an emotion category, and its size relative to the whole pie indicates the proportion of samples with that emotion. It offers a concise summary of emotion distribution.</li>
                <li><b>Fig16<br/>Class Distribution Before Balancing :</b> This bar plot displays the distribution of different emotion categories in the dataset before applying any balancing techniques. Each bar represents an emotion, and its height indicates the number of samples with that emotion. It helps identify class imbalances.</li>
                <li><b>Fig17<br/>Class Distribution After Balancing :</b> Similar to Fig12, this bar plot shows the distribution of emotion categories in the dataset after applying the RandomOverSampler to balance the classes. Balancing ensures an equal representation of different emotions, reducing class imbalance issues.</li>
                <li><b>Fig18<br/>Distribution of Labels after Splitting :</b> This plot visualizes the distribution of emotion labels after splitting the dataset into training, validation, and testing sets. It shows the frequency of each emotion label across different subsets of the data.</li>
            </ul>
        
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
