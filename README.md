# Enterprise Customer Intelligence Platform

An end-to-end Customer Intelligence platform combining Machine Learning,
Customer Segmentation, RFM Analysis, SQL Analytics, FastAPI and Generative AI
with Retrieval-Augmented Generation (RAG).

---

## Project Overview

This project demonstrates how machine learning and generative AI can be
combined to support customer-focused business decisions.

The platform provides:

- Customer churn prediction
- Customer segmentation using K-Means
- RFM analysis
- Statistical analysis and hypothesis testing
- SQL-based customer analytics
- REST API for churn prediction
- RAG-based business knowledge assistant
- Interactive Streamlit dashboard

The dataset used in this project is synthetic and was created for
demonstration and learning purposes.

---

## Business Problem

Businesses need to identify customers who may churn, understand customer
behavior, identify high-value customers and provide actionable insights.

This platform addresses these requirements by combining:

1. Predictive Machine Learning
2. Customer Segmentation
3. RFM Analytics
4. SQL Analytics
5. Generative AI and RAG

---

## Architecture

```text
                    ENTERPRISE CUSTOMER
                       INTELLIGENCE
                            |
          +-----------------+-----------------+
          |                                   |
          v                                   v
    ML / Analytics                         GenAI / RAG
          |                                   |
    Churn Prediction                    Business Documents
          |                                   |
    Feature Engineering                    Chunking
          |                                   |
    Model Evaluation                     Embeddings
          |                                   |
    Threshold Tuning                     ChromaDB
          |                                   |
    Customer Segmentation                  Groq LLM
          |                                   |
    RFM Analysis                            |
          |                                   |
    SQL Analytics                           |
          |                                   |
          +-----------------+-----------------+
                            |
                            v
                    Streamlit Dashboard
```

Technology Stack
Programming
Python
SQL
Data Processing
Pandas
NumPy
SciPy
Machine Learning
Scikit-learn
XGBoost
Logistic Regression
Random Forest
K-Means Clustering
Generative AI
Groq
Sentence Transformers
ChromaDB
Retrieval-Augmented Generation (RAG)
API
FastAPI
Uvicorn
Dashboard
Streamlit
Plotly
Database
SQLite
SQLAlchemy
Development
Git
GitHub
VS Code

Project Structure
Enterprise_Customer_Intelligence/
|
├── app/
│ ├── **init**.py
│ └── dashboard.py
│
├── data/
│ ├── knowledge_base/
│ │ ├── customer_retention_policy.txt
│ │ ├── customer_support_policy.txt
│ │ └── customer_segmentation_guide.txt
│ │
│ ├── processed/
│ └── raw/
│
├── notebooks/
│ └── 01_eda_statistical_analysis.ipynb
│
├── src/
│ ├── data/
│ │ ├── generate_data.py
│ │ ├── data_loader.py
│ │ └── preprocessing.py
│ │
│ ├── database/
│ │ ├── database.py
│ │ └── sql_analysis.py
│ │
│ ├── features/
│ │ ├── feature_engineering.py
│ │ ├── rfm_segmentation.py
│ │ └── segment_profiling.py
│ │
│ ├── models/
│ │ ├── train_model.py
│ │ ├── threshold_tuning.py
│ │ ├── save_model.py
│ │ ├── predict.py
│ │ └── customer_segmentation.py
│ │
│ └── rag/
│ ├── document_loader.py
│ ├── vector_store.py
│ └── rag_pipeline.py
│
├── tests/
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt

1. Data Generation
   A synthetic customer dataset containing 5,000 customer records was
   generated
   The dataset contains:

Customer ID
Age
Tenure
Monthly Charges
Total Purchases
Average Transaction Value
Support Tickets
Churn

Missing values were intentionally introduced into the dataset to demonstrate
data preprocessing.
