import streamlit as st

def run():
   
    # Title
    st.markdown(
        """
        <div style='border: 4px solid #447ECC; border-radius: 15px; padding: 20px; margin-bottom: 30px; background-color:#1B1F31 ;'>
            <h1 style='text-align: center; color: #E8D9CF;'>Comparative Study</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Define image paths, captions, and descriptions
    graph_info = [
        {"path": "Visualization/Fig25.png",
         "caption": "Fig 1.Accuracy of all models",
         "description": """
            <ul>
                <h4><b>Accuracy of All Models:</b></h4>
                <li>This plot illustrates the accuracy achieved by each model in classifying emotions from text data in our project.</li>
                <li>Accuracy indicates the percentage of correctly classified instances out of the total instances.</li>
                <li>Higher accuracy values suggest better overall performance of the models in correctly predicting emotions.</li>
            </ul>
         """},
        {"path": "Visualization/Fig26.png",
         "caption": "Fig 2.Precision of all models",
         "description": """
            <ul>
                <h4><b>Precision of All Models:</b></h4>
                <li>In our project context, precision measures how precise each model is in identifying specific emotions.</li>
                <li>This plot displays the precision score for each model, reflecting the proportion of correctly predicted instances of a particular emotion out of all instances predicted as that emotion.</li>
                <li>Higher precision values imply that the model makes fewer false positive predictions for emotions.</li>
            </ul>
         """},
        {"path": "Visualization/Fig27.png",
         "caption": "Fig 3.F1 score of all models",
         "description": """
            <ul>
                <h4><b>F1-score of All Models:</b></h4>
                <li>F1-score, being the harmonic mean of precision and recall, provides a balanced measure of a model's performance in our emotion classification project.</li>
                <li>It considers both false positives and false negatives, making it suitable for evaluating models in scenarios where class imbalance exists.</li>
                <li>This plot showcases the F1-score for each model, offering insights into their overall effectiveness in correctly classifying emotions.</li>
            </ul>
         """},
        {"path": "Visualization/Fig28.png",
         "caption": "Fig 4.Recall of all models",
         "description": """
            <ul>
                <h4><b>Recall of All Models:</b><h4>
                <li>Recall, or sensitivity, assesses each model's ability to capture all instances of a specific emotion from the dataset.</li>
                <li>In our project, recall indicates the proportion of correctly identified instances of a particular emotion out of all instances of that emotion in the dataset.</li>
                <li>Higher recall values suggest that the model can effectively identify a larger portion of instances belonging to a specific emotion class.</li>
            </ul>
         """}
    ]

    # Display images and descriptions
    for info in graph_info:
      col1, col2 = st.columns([5, 3])  # Divide the layout into 3/4 and 1/4 columns
      with col1:
            st.image(info["path"], caption=info["caption"], width=None, use_column_width="auto")
            st.markdown("<style>img {border-radius: 10px;}</style>", unsafe_allow_html=True)
      with col2:
            st.markdown(info["description"], unsafe_allow_html=True)
      

    # Conclusion box
    st.markdown(
        """
        <div style='border: 4px solid #447ECC; border-radius: 15px; padding: 20px; margin-top: 30px; background-color:#1B1F31 ;'>
            <h2 style='text-align: left; color:#E8D9CF ;'>Conclusion</h2>
            <p style='color: ; color:#E8D9CF ;padding-left: 20px;'>The LSTM model exhibited superior performance in accuracy, precision, F1-score, and recall compared to Naive Bayes, Random Forest, Gradient Boosting, and SVM models for emotion classification. Its robustness in capturing diverse emotions from text data makes it the optimal choice despite potential considerations like computational complexity and training time.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

run()
