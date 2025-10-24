
# 🎯 WISE-AIML 2025 Capstone Project: Amazon Reviews Sentiment Analysis

## 1\. Executive Summary

This project implements an efficient and highly accurate machine learning pipeline to classify customer sentiment from a large-scale dataset of $\mathbf{4}$ million Amazon reviews. By comparing several models, we selected an optimal solution that surpassed the project benchmark.

| Metric | Result | Target Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **Final F1-Score** | **$\mathbf{0.9182}$** | $0.9160$ | **SUCCESS: Exceeded Target** |
| **Best Model** | **Logistic Regression Pipeline** | N/A | **Selected for Deployment** |

-----

## 2\. Approach and Methodology

The core goal was **Binary Sentiment Classification**: classifying reviews (4/5 stars) as **Positive** and (1/2 stars) as **Negative**.

### 2.1. Project Architecture (Tech Stack Used)

  * **Core Language:** Python
  * **Classification:** Scikit-learn (Logistic Regression, Multinomial Naive Bayes)
  * **Benchmark:** FastText (Subword Embeddings)
  * **Feature Engineering:** TF-IDF Vectorization
  * **Deployment/Demo:** Streamlit, Joblib, Pyngrok
  * **Version Control:** Git / GitHub

### 2.2. Feature Engineering Rationale

The final model relies on a **TF-IDF Pipeline**. This method was chosen for its high efficiency with sparse, high-dimensional text data:

  * **Term Frequency-Inverse Document Frequency (TF-IDF):** Weights words based on their relevance (not just frequency), effectively filtering out common, uninformative words.
  * **N-grams:** We utilized **Unigrams and Bigrams ($\text{ngram\_range}=(1, 2)$)** to capture short phrases like "not good" or "very disappointed," which is essential for accurate sentiment detection.

### 2.3. Final Model Comparison

Our comparison established **Logistic Regression** as the superior and most stable final model.

| Model | Feature Approach | Precision | Recall | **F1-Score** |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | TF-IDF (1,2-grams) | $0.9148$ | $0.9215$ | **$\mathbf{0.9182}$** |
| **FastText (Benchmark)** | Subword Embeddings | $0.9160$ | $0.9160$ | $0.9160$ |
| **Multinomial Naive Bayes**| TF-IDF (1,2-grams) | $0.8726$ | $0.8806$ | $0.8766$ |

-----

## 3\. Project Deployment and Live Demo

The final Logistic Regression model is saved as a complete pipeline (`.pkl`) and deployed via a simple **Streamlit** web application for demonstration.

### 3.1. Repository Structure

```
WISE-AIML-2025-Amazon-Sentiment-Analysis/
├── data/                    # (Contains large files excluded via .gitignore)
├── models/
│   ├── amazon_sentiment_lr_pipeline.pkl  # <-- FINAL DEPLOYED MODEL
│   └── fasttext_model_amzn.bin
├── notebooks/
│   └── Amazon_Sentiment_Capstone.ipynb   # <-- Full EDA, Training, and Analysis
├── src/
│   └── app.py                      # <-- Streamlit Application Code
├── .gitignore
└── requirements.txt
```

### 3.2. Setup and Usage Instructions

To run the live Streamlit demo:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/WISE-AIML-2025-Amazon-Sentiment-Analysis.git
    cd WISE-AIML-2025-Amazon-Sentiment-Analysis
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Authentication:** (Set your ngrok token for a stable tunnel)
    ```bash
    # Note: Replace <YOUR_AUTH_TOKEN> with your actual ngrok token
    !ngrok config add-authtoken <YOUR_AUTH_TOKEN>
    ```
4.  **Launch the Demo:** Run the final launch script cell located at the end of the `Amazon_Sentiment_Capstone.ipynb` notebook. This will output a public URL you can click to access the live web application.

-----

## 4\. Evaluation and Repository Stats

This repository is maintained to demonstrate quality and complete project lifecycle adherence:

  * **Code Walkthrough:** The **`notebooks/Amazon_Sentiment_Capstone.ipynb`** provides a sequential, documented, and runnable record of all steps from EDA to final evaluation.
  * **Git Repository Stats (Commits):** The commit history tracks progress clearly across all phases: `INIT` $\rightarrow$ `EDA` $\rightarrow$ `MODEL` $\rightarrow$ `DEPLOY`.
  * **Code Quality:** The pipeline architecture ensures a clean separation of concerns and robust code structure.