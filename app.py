import streamlit as st
import pickle
import numpy as np
import gensim
from gensim.models import KeyedVectors

# Load pretrained Word2Vec model (Google News Word2Vec)
try:
    model_path = 'path/to/your/pretrained/GoogleNews-vectors-negative300.bin'  # Update with the path to your model file
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
except FileNotFoundError:
    st.error("Pretrained Word2Vec model file not found. Ensure it's in the correct path.")
    st.stop()

# Load XGBoost model
try:
    with open('model/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
except FileNotFoundError:
    st.error("XGBoost model file not found. Ensure 'xgboost_model.pkl' is in the 'model/' directory.")
    st.stop()

# Function to preprocess questions
def preprocess_questions(q1, q2):
    q1_tokens = str(q1).split()
    q2_tokens = str(q2).split()
    return q1_tokens, q2_tokens

# Function to compute embeddings using the pretrained Word2Vec model
def sentence_to_vec(tokens, model):
    words = [word for word in tokens if word in model.wv]
    if len(words) == 0:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in words], axis=0)

def generate_embeddings(q1_tokens, q2_tokens, model):
    q1_vec = sentence_to_vec(q1_tokens, model)
    q2_vec = sentence_to_vec(q2_tokens, model)
    features = np.concatenate((q1_vec, q2_vec))
    return features

# Function for prediction
def predict_similarity(q1, q2):
    q1_tokens, q2_tokens = preprocess_questions(q1, q2)
    features = generate_embeddings(q1_tokens, q2_tokens, w2v_model)
    prediction = xgb_model.predict([features])[0]
    probability = xgb_model.predict_proba([features])[0]
    return prediction, probability

# Streamlit App
st.title("Question Similarity Checker")
st.write("Enter two questions to check if they are similar or not.")

q1 = st.text_input("Question 1:")
q2 = st.text_input("Question 2:")

if st.button("Check Similarity"):
    if q1.strip() and q2.strip():
        try:
            prediction, probability = predict_similarity(q1, q2)
            st.write(f"Prediction: {'Duplicate' if prediction == 1 else 'Not Duplicate'}")
            st.write(f"Confidence: {probability[prediction]:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter both questions before checking similarity.")
