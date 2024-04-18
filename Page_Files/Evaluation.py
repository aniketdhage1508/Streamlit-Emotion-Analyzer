import streamlit as st

# Image descriptions for each model
model_names = ["1. LSTM", "2. NAIVE BAYES", "3. RANDOM FOREST", "4. GRADIENT BOOSTING", "5. SVM"]

# Corresponding descriptions for each model
model_descriptions = [
    """
    <ul style='color:white;'>
        <li>Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture.</li>
        <li>It's widely used for sequence prediction tasks due to its ability to retain long-term dependencies.</li>
        <li>In this evaluation, the LSTM model is trained and evaluated for emotion classification based on text data.</li>
    </ul>
    """,
    """
    <ul style='color:white;'>
        <li>Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with an assumption of independence between features.
         </li>
        <li>Despite its simplicity, it's often used as a baseline for text classification tasks and works well with high-dimensional data.</li>
    </ul>
    """,
    """
    <ul style='color:white;'>
        <li>Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes as the prediction.
        </li>  
        <li>It's known for its robustness and ability to handle high-dimensional data with ease.</li>
        <li>Random Forests are suitable for both classification and regression tasks.</li>
    </ul>
    """,
    """
    <ul style='color:white;'>
        <li>Gradient Boosting is another ensemble learning technique that builds decision trees sequentially, where each tree corrects the errors of the previous one.</li>
        <li>It's known for its high predictive accuracy and is particularly effective in handling heterogeneous features and large datasets.</li>
        <li>Gradient Boosting models are often used in classification and regression problems.</li>
    </ul>
    """,
    """
     <ul style='color:white;'>
        <li>SVM is a supervised machine learning algorithm used for classification and regression tasks.</li>
        <li>It works by finding the hyperplane that best divides a dataset into classes while maximizing the margin between the classes.</li>
        <li>SVM is effective in high-dimensional spaces and is versatile in handling various types of data through the use of different kernel functions.</li>
    </ul>
    """
]

# Corresponding paths to images for each model
model_image_paths = [
    ["Visualizations/Fig15.png", "Visualizations/Fig16.png"],
    ["Visualizations/Fig17.png", "Visualizations/Fig18.png"],
    ["Visualizations/Fig19.png", "Visualizations/Fig20.png"],
    ["Visualizations/Fig21.png", "Visualizations/Fig22.png"],
    ["Visualizations/Fig23.png", "Visualizations/Fig24.png"]
]

# Main function to display the images and descriptions
def run():
    #title s
    st.markdown(
    """
    <div style='border: 4px solid #447ECC; border-radius: 15px; padding: 20px; margin-bottom: 30px; background-color:#1B1F31 ;'>
        <h1 style='text-align: center; color: #E8D9CF;'>Model Insight Showcase</h1>
    </div>
    """,
    unsafe_allow_html=True
)
    
    # Display images for each model along with their descriptions
    for model_name, model_description, model_images in zip(model_names, model_descriptions, model_image_paths):
        st.header(model_name)
        st.write(f"<span style='color: #656B79;'>{model_description}</span>", unsafe_allow_html=True)
        
        # Display images in wide mode with two images per row
        for i in range(0, len(model_images), 2):
            col1, col2 = st.columns(2)
            with col1:
                st.image(model_images[i], caption='Confusion Matrix', use_column_width=True)
                st.markdown("<style>img {border-radius: 10px;}</style>", unsafe_allow_html=True)
            with col2:
                if i + 1 < len(model_images):
                    st.image(model_images[i + 1], caption='Evaluation Metrices', use_column_width=True)
                    st.markdown("<style>img {border-radius: 10px;}</style>", unsafe_allow_html=True)
                
        st.write("---")

if __name__ == "__main__":
    main()