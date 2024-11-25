# **IMDB Search Engine**
<img src="./IMDB_Logo.jpeg" alt="IMDb Logo" width="100%" height="auto" />

This repository contains the implementation of a comprehensive IMDb search engine. The project employs a variety of techniques in information retrieval, classification, and recommendation systems, developed across three distinct phases.

> **Note**: This repository includes implementation code but excludes datasets and model parameters due to their large size.

---

## **Table of Contents**
- [Overview](#overview)
- [Phases](#phases)
  - [Phase 1: Traditional Search Engine Implementation](#phase-1-traditional-search-engine-implementation)
  - [Phase 2: Enhanced Search Results and Content Classification](#phase-2-enhanced-search-results-and-content-classification)
  - [Phase 3: RAG, BERT, and Recommender Systems](#phase-3-rag-bert-and-recommender-systems)
- [Key Tools and Libraries Used](#key-tools-and-libraries-used)
- [Example Results](#example-results)
- [Limitations](#limitations)

---

## **Overview**
The IMDb search engine project progresses through three phases:

1. **Phase 1**: Implementation of traditional search engine techniques.
2. **Phase 2**: Enhanced search capabilities using link analysis, embeddings, and classification.
3. **Phase 3**: Independent tools for Retrieval-Augmented Generation (RAG), recommender systems, and BERT-based classification.

---

## **Phases**

### **Phase 1: Traditional Search Engine Implementation**
This phase establishes the foundation of the search engine using classical information retrieval methods.

#### Data Collection
Data was collected using a custom web crawler built with **BeautifulSoup** and **Requests**. The process involved:
- Navigating IMDb pages.
- Parsing metadata such as titles, genres, ratings, directors, cast, budgets, and links.
- Storing structured data in a format suitable for retrieval.

Challenges:
- Handling IMDb's rate limits to avoid blocking.
- Ensuring data consistency across pages with varying formats.

#### Search Algorithms
Implemented the following search algorithms:
- **Vector Space Model**
- **Okapi BM25**
- **Unigram Language Model**

#### Duplicate Detection
Efficiently detected duplicate documents using **LSH MinHash**, reducing computational overhead to O(n).

#### Spell Correction
Implemented a spell correction system that:
- Identifies misspellings in user queries by comparing against a vocabulary built from the crawled data.
- Generates candidates using a dictionary-based lookup.
- Ranks candidates using **Jaccard similarity**, a metric that measures overlap between query terms and candidate terms, adjusted by their term frequency.

#### Evaluation
Assessed search results using standard retrieval metrics like precision, recall, and F1-score. Snippets were extracted to provide users with contextually relevant excerpts for each query.

---

### **Phase 2: Enhanced Search Results and Content Classification**
Building on Phase 1, this phase focuses on refining search results and classifying content more effectively.

#### Link Analysis
Enhanced result ranking by applying the **HITS algorithm**, which calculates:
- **Hub Scores**: Reflecting the quality of outgoing links.
- **Authority Scores**: Indicating the importance of a document based on incoming links.

This recursive relationship improved the relevance of search results.

#### Text Embeddings
Trained a **FastText model** (Skip-gram) to generate word embeddings. These embeddings:
- Capture semantic relationships between words, even if they don't match exactly.
- Enable the system to relate similar terms like "cinema," "movie," and "film."

#### Content Classification
Classified movie metadata into genres and categories using:
- Support Vector Machine (SVM)
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Deep Learning Model

#### Clustering
Applied clustering algorithms on text embeddings to group similar movies and discover hidden patterns in the data.

---

### **Phase 3: RAG, BERT, and Recommender Systems**
This phase adds advanced tools and features, implemented as standalone modules.

#### Retrieval-Augmented Generation (RAG)
Combined document retrieval with language model generation using **LangChain** and Hugging Face models:
- Retrieved top 10 relevant documents based on user queries.
- Used these documents as context for generating factually accurate and context-aware responses.
- Expanded user prompts with an additional LLM before processing, enhancing query precision.

#### Recommender System
Built a recommendation engine using multiple approaches:
- **Content-Based Filtering**
- **Collaborative Filtering**
- **PCA (Principal Component Analysis)** for dimensionality reduction.

Predicted user movie ratings based on historical preferences using tools like `Surprise`.

#### BERT for Genre Classification
Fine-tuned a **BERT model** using the **Masked Language Modeling (MLM)** task to improve the accuracy of genre predictions.

---

## **Key Tools and Libraries Used**
- **Data Crawling**: `BeautifulSoup`, `Requests`
- **Search Engine Models**: `NumPy`, `Scikit-learn`
- **Text Embeddings**: `FastText`
- **RAG and LLMs**: `LangChain`, `Hugging Face Transformers`
- **Recommender System**: `Pandas`, `Surprise`
- **Deep Learning**: `PyTorch`, `TensorFlow`

---

## **Example Results**
Here’s a sample output from the final search engine. Note that Phase 3 tools (RAG, LLM, and Recommender System) are not integrated into this result and remain independent:

[IMDb Search Results Demo](https://github.com/user-attachments/assets/61861608-44c3-4efc-acba-e269c91c8e2b)

(Values next to movie titles indicate their rounded scores.)

---

## **Limitations**
While datasets and model parameters are not included due to size constraints, users can replicate the results by:
1. Crawling IMDb data using tools like `BeautifulSoup` or similar web scraping frameworks.
2. Training models using libraries like `FastText`, `Scikit-learn`, or `Hugging Face Transformers`.

Refer to the respective sections for details on data preprocessing and model training steps.
