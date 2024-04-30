import streamlit as st
import pandas as pd
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load the trained classifier
loaded_clf = load('cv_classifier.joblib')

# Load the TfidfVectorizer and PCA for preprocessing
tfidf_model = load('tfidf_model.joblib')
pca = load('pca_model.joblib')

# Mapping of category codes to category names
label_to_category = {
    0: 'BUSINESS-DEVELOPMENT',
    1: 'DATA-SCIENCE',
    2: 'ENGINEERING',
    3: 'DESIGNER',
    # Add more mappings as needed
}

# Define function for prediction
def predict_category(text):
    # Preprocess the new data using the same TfidfVectorizer and PCA
    new_embeddings = tfidf_model.transform([text])
    new_embeddings = pca.transform(new_embeddings.toarray())

    # Make predictions on the new data using the loaded model
    predictions = loaded_clf.predict(new_embeddings)

    # Map the predicted category codes to category names
    predicted_categories = [label_to_category.get(code, 'UNKNOWN') for code in predictions]

    return predicted_categories

# Main Streamlit app
def main():
    st.title("CV Analyzer")

    # Text input for CV
    cv_text = st.text_area("Paste your CV text here:")

    if st.button("Analyze"):
        if cv_text:
            categories = predict_category(cv_text)
            st.write("Predicted Categories:")
            for category in categories:
                st.write("- " + category)
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
