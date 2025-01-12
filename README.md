# Quora Question Pairs Similarity Checker

This project is part of the [Kaggle Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) competition. The task is to determine whether two questions asked on Quora are semantically similar or not. The goal is to predict whether a pair of questions are asking the same thing, which is useful for duplicate question detection.

## Problem Statement

Given two questions, the goal is to identify if they are asking the same question or not. For example:

- "What is the capital of France?" and "What is the capital of Paris?" are considered **similar**.
- "What is the capital of France?" and "How does a plane fly?" are considered **not similar**.

## Approach 
  
In this project, we use a combination of **text preprocessing**, **Word2Vec embeddings**, and **XGBoost** to classify the similarity between pairs of questions.

### Steps Taken:

1. **Text Preprocessing**:
   - The first step is to clean and preprocess the text data by tokenizing the input questions and removing any unnecessary characters (e.g., punctuation, stopwords). Tokenization is essential for converting text into words that can be used for embedding generation.

2. **Word2Vec Embeddings**:
   - Word2Vec is used to convert words into dense vector representations, capturing semantic meaning. By averaging the vectors of words in a question, we generate a sentence embedding that represents the meaning of the question.
   - **Pretrained Word2Vec Model**: Instead of training our own Word2Vec model, we use a pretrained Word2Vec model (Google News Word2Vec) to save time and resources. The pretrained model contains word embeddings learned from a vast corpus of text (Google News), providing rich semantic representations for a wide range of words.

3. **XGBoost Classifier**:
   - After generating the embeddings, we concatenate the sentence embeddings for both questions in a pair to create a feature vector.
   - XGBoost is then used to classify whether the pair of questions are similar or not. XGBoost is an efficient and powerful gradient boosting algorithm, known for its speed and performance in structured/tabular data.

### Why These Techniques?

- **Text Preprocessing**: Necessary to prepare raw text data for machine learning models. Tokenization helps to break text into manageable parts (words), while other preprocessing steps (e.g., lowercasing, removing stopwords) reduce noise in the data.
  
- **Word2Vec Embeddings**: Word2Vec, a neural network-based technique, generates vector representations for words based on their context in a large corpus. It captures semantic relationships between words, which helps in measuring the similarity between questions. Pretrained Word2Vec models like Google News Word2Vec are extremely useful because they have already been trained on a large and diverse corpus, allowing us to use high-quality embeddings without having to train the model ourselves.

- **XGBoost**: XGBoost is a popular machine learning algorithm, particularly for classification tasks. It is effective in handling large datasets and works well with a variety of features. The powerful feature importance and regularization capabilities make XGBoost a great choice for this task.
