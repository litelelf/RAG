
# Retrieval-Augmented Generation (RAG) Demo with Streamlit

This project demonstrates a Retrieval-Augmented Generation (RAG) system using Streamlit, which combines information retrieval and language generation to answer user queries based on a dataset. 
The system implements the following features:


1. **Data Collection**: Loads 50 articles from the SQuAD dataset (`train-v1.1.json`)
2. **Document Retrieval**: Implements a retriever model using both semantic search and TF-IDF
3. **Language Generation**: Integrates a Large Language Model (LLM) to generate answers based on retrieved documents
4. **Evaluation**: Calculates the Precision@10 metric and response time percentiles (q50, q90, q99)
5. **Deployment**: Provides an interactive interface using Streamlit

---

## Table of Contents

- [Installation](#installation)
- [Project Overview](#project-overview)
  - [1. Data Collection](#1-data-collection)
  - [2. Document Retrieval](#2-document-retrieval)
  - [3. Language Generation](#3-language-generation)
  - [4. Evaluation](#4-evaluation)
  - [5. Deployment](#5-deployment)
- [Usage](#usage)
- [Notes](#notes)

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/rag-demo.git
   cd rag-demo
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Hugging Face API token:**

    - Replace HUGGINGFACE_API_TOKEN in the project with your token
        

4. **Ensure the `train-v1.1.json` file is in the project directory.**

---

## Project Overview

### 1. Data Collection

- **Process:**

  - Opens and reads the `train-v1.1.json` file
  - Iterates over the articles, extracts the title, context, and Q&A pairs
  - Limits the number of articles to 50
  - Stores the data in a list of dictionaries, each containing:
    - `title`
    - `content` (context)
    - `qas` (questions and answers)

### 2. Document Retrieval

#### 2.1 Using a Hugging Face Model

- **Model Used:** `SentenceTransformer('all-MiniLM-L6-v2')`
- **Purpose:** Converts documents and queries into embeddings for semantic similarity comparison

#### 2.2 Keyword and TF-IDF Search

- **KeyBERT:**

  - Used to extract keywords from the query
  - 
- **TF-IDF Vectorizer:**

  - Transforms documents into TF-IDF features
  - **Function Used:** `retrieve_tfidf()`
  - Calculates cosine similarity between the query vector and document vectors

### 3. Language Generation

#### 3.1 Integrating an LLM into the Answer Generation Pipeline

- **LLM Used:** `meta-llama/lama-2-7b-chat-hf` accessed via Hugging Face Inference API
- **Function Used:** `generate_answer()`

- **Process:**

  - Concatenates the content of retrieved documents to form the context
  - Limits the context length to prevent exceeding model input limits
  - Forms a prompt combining the query and context
  - Sends the prompt to the LLM to generate an answer

#### 3.2 Analyzing and Selecting the Optimal Number of Candidates for the LLM

- **User Interaction:**

  - The number of documents (`top_k`) to retrieve can be adjusted via a slider in the Streamlit interface
  - Users can experiment with different values to find the optimal number

### 4. Evaluation

#### Calculating Precision@10

- **Function Used:** `compute_precision_at_k()`
- **Process:**

  - Iterates over all questions in the dataset
  - Retrieves top `k` documents for each question
  - Checks if the correct answer exists within the retrieved documents
  - Calculates Precision@k for each question and computes the average

### 5. Deployment

#### Deploying the Solution in Streamlit

- The entire application is built using Streamlit, providing an interactive web interface
- Users can input queries, select retrieval methods, and view generated answers and evaluation metrics

***

## Usage

1. **Run the Streamlit App:**

   ```bash
   streamlit run pj.py
   ```

2. **Interact with the App:**

   - Enter your question in the input field
   - Select the retrieval method (Semantic Search or TF-IDF)
   - Adjust the number of documents to retrieve using the slider
   - Click on **"Вычислить Precision@10"** to calculate the Precision@10 metric
   - Click on **"Вычислить квантили времени ответа"** to view response time percentiles

***

## Notes
  Response times may vary based on the model and network latency. 
  Limiting the context length and the number of retrieved documents can improve performance