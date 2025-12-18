# Interpretable Machine Learning for Fraud Detection in Large-Scale Financial Transactions

> **Note:** This repository accompanies the research paper submitted to **arXiv**.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![Research](https://img.shields.io/badge/Focus-Academic%20Research-red.svg)

## Abstract
This project presents a research-oriented and interpretable machine learning framework for detecting fraudulent financial transactions in large-scale banking data. Using a dataset of approximately **18 million transactions**, the system emphasizes behavioral feature engineering, probabilistic risk estimation, and post-hoc analytical evaluation. An interactive experimental dashboard is provided to support controlled transaction testing, qualitative error analysis, and interpretability-driven inspection. This project is designed as an academic research artifact, suitable for MSc applications and future peer-reviewed publication.

---

## 1. Research Motivation
Fraud detection in financial systems presents multiple real-world and research challenges:
* **Extreme class imbalance:** Fraud cases represent less than 0.2% of the data.
* **High cost of false negatives:** Leading to direct financial loss.
* **Regulatory demand for interpretability:** Modern systems require explainable decision-making.

**This work focuses on:**
* Modeling transactional behavior rather than customer identity.
* Combining statistical learning with domain-driven financial indicators.
* Producing transparent risk outputs instead of opaque binary decisions.

---

## 2. Dataset Description
* **Source:** Financial Transactions Dataset (AIML – Synthetic & Anonymized).
* **Scale:** ~18,000,000 transactions.
* **Fraud Ratio:** ~0.17%.
* **Transaction Types:** `TRANSFER`, `CASH_OUT`, `PAYMENT`, `DEBIT`, `CASH_IN`.
* **Ethical Note:** Contains no Personal Identifiable Information (PII).

---

## 3. System Overview & Interactive Evaluation

### 3.1 Transaction Input Interface
The system provides an interactive interface that allows manual transaction simulation, enabling controlled experimentation with transaction attributes.

![Transaction Input](photos/1.png)

### 3.2 Prediction Output & Risk Interpretation
For each evaluated transaction, the system outputs a Fraud Probability, Aggregated Risk Score, and Behavioral Alerts.

![Risk Interpretation](photos/2.png)

---

## 4. Exploratory Data & Model Behavior Analysis

### 4.1 Fraud Probability Distribution
The distribution highlights how the model allocates probability mass under extreme class imbalance.

![Probability Distribution](photos/3.png)

### 4.2 Transaction Amount vs Fraud Label
A log-scaled comparison demonstrates separation trends between legitimate and fraudulent transactions.

![Amount vs Label](photos/4.png)

### 4.3 Transaction Type Analysis
Analysis of fraud incidence across different operation types (notably higher in `TRANSFER` and `CASH_OUT`).

![Transaction Type Distribution](photos/5.png)
![Fraud Rate by Type](photos/6.png)

### 4.4 Feature Correlation Analysis
Highlights relationships between transaction attributes and engineered behavioral features.

![Correlation Analysis](photos/7.png)

---

## 5. Model Behavior & Error Analysis

### 5.1 Amount vs Predicted Fraud Probability
This visualization reveals the concentration of predicted risk in specific transaction regimes.

![Amount vs Prediction](photos/8.png)

### 5.2 Top Suspicious Transactions
A ranked view of high-risk transactions supports manual audit and qualitative error analysis.

![Top Suspicious](photos/9.png)

---

## 6. Methodology Summary
* **Model:** Logistic Regression (Interpretable baseline).
* **Preprocessing:** Standardization and One-hot encoding.
* **Feature Engineering:**
    * Balance differentials.
    * Amount-to-balance ratios.
    * Behavioral risk flags.

![Methodology Workflow](photos/10.png)

---

## 7. Research Value
This project demonstrates applied machine learning in a realistic financial setting, focusing on:
* Handling highly imbalanced datasets.
* Behavioral feature engineering grounded in domain logic.
* Reproducible and inspectable experimental analysis.

**Suitable As:** MSc research portfolio material or a foundation for peer-reviewed research.

---

## 8. Limitations & Future Work
* **Temporal modeling:** Using sequence-based approaches (LSTM / Transformers).
* **Explainable Ensembles:** Integrating SHAP or LIME with Gradient Boosting.
* **Real-time deployment:** Implementing streaming inference pipelines.

---

## Author
**Mariam Zakaria**
*MSc Applicant — Machine Learning & Data Science*
*Research Interests: Fraud Detection, Interpretable Machine Learning, Applied AI Systems.*

