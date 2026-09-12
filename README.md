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

2. Data Preprocessing

The preprocessing pipeline performs:

Duplicate detection
Missing-value handling
Data validation
Median imputation for missing average transaction values

The processed dataset is saved under:
data/processed/


3. Feature Engineering

Additional business features were created:

estimated_lifetime_value
estimated_total_spending
support_ticket_rate
purchase_frequency

These features are used for customer analysis and machine learning.

4. Exploratory Data Analysis

Statistical analysis was performed using Pandas, NumPy, Matplotlib and
SciPy.

The analysis includes:

Churn distribution
Feature correlations
Churned vs non-churned customer comparison
Hypothesis testing
Support-ticket analysis
Tenure analysis
Statistical Findings

The synthetic dataset showed:

Churn rate of approximately 14.66%
Churned customers had lower average tenure than non-churned customers
Churned customers had a higher average number of support tickets
The tenure difference was statistically significant
The support-ticket difference was statistically significant

Correlation was used as an association measure and not interpreted as
causation.

5. Churn Prediction

Three classification models were compared:

Logistic Regression
Random Forest
XGBoost

Because the dataset contains class imbalance, accuracy was not treated as
the only evaluation metric.

The following metrics were evaluated:

Precision
Recall
F1 Score
ROC-AUC
Model Comparison
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.585	0.195	0.585	0.293	0.624
Random Forest	0.826	0.286	0.122	0.171	0.595
XGBoost	0.721	0.220	0.354	0.272	0.598

For this synthetic dataset, Logistic Regression provided the strongest
ROC-AUC, recall and F1 score among the tested models.

6. Threshold Tuning

The default classification threshold of 0.50 was evaluated against several
alternative thresholds.

A threshold of 0.45 provided a better balance for the project's churn
detection objective.

At threshold 0.45:

Precision: 0.186
Recall:    0.782
F1 Score:  0.301
ROC-AUC:   0.624

The threshold was lowered because missing a potentially churn-prone
customer can be costly from a retention perspective.

The threshold should ultimately be selected according to the business cost
of false positives and false negatives.

7. Customer Segmentation

RFM analysis was used to describe customer purchasing behavior.

RFM consists of:

Recency
Frequency
Monetary

Because the synthetic dataset does not contain transaction timestamps,
recency was implemented as a tenure-derived proxy.

K-Means clustering was then applied to the RFM features.

Different values of K were evaluated using the silhouette score.

The best result in this dataset was:

K = 2
Silhouette Score = 0.3493

The resulting segments were profiled according to:

Customer count
Recency
Purchase frequency
Monetary value
Churn rate

The higher-value segment had substantially higher purchase frequency and
monetary value.

Segmentation is treated as behavioral grouping and not as a direct churn
prediction model.

8. SQL Analytics

Customer data was loaded into SQLite using SQLAlchemy.

The SQL analysis includes:

Total customer count
Churn distribution
Overall churn rate
Average monthly charges by churn status
Average support tickets by churn status
High-support-ticket customers
Rule-based customer risk classification
Churn rate by support-ticket count
High-value churned customers
CTE-based high-risk customer analysis
Lifetime-value ranking using window functions

SQL concepts demonstrated include:

GROUP BY
CASE WHEN
HAVING
Subqueries
Common Table Expressions (CTEs)
Window functions
RANK
9. FastAPI Prediction Service

A REST API was created using FastAPI.

Endpoint
POST /predict

The API accepts customer information such as:

age
tenure_months
monthly_charges
total_purchases
avg_transaction_value
support_tickets

The API returns:

Churn probability
Churn prediction
Risk level
Classification threshold

Swagger documentation is available through:

/docs

when the API is running locally.

10. Generative AI and RAG

A Retrieval-Augmented Generation pipeline was implemented for business
knowledge questions.

The knowledge base contains documents covering:

Customer retention policy
Customer support policy
Customer segmentation guide
RAG Pipeline
Business Documents
        |
        v
Document Loading
        |
        v
Text Chunking
        |
        v
Sentence Transformer
        |
        v
Embeddings
        |
        v
ChromaDB
        |
        v
Semantic Retrieval
        |
        v
Relevant Context
        |
        v
Groq LLM
        |
        v
Grounded Business Answer
        |
        v
Source Attribution

The system retrieves the most relevant chunks from the knowledge base and
passes them as context to the LLM.

The prompt instructs the model to:

Use only the provided context
Avoid inventing facts
State when information is unavailable
Provide concise business-focused answers

Source documents and chunk IDs are also returned with the generated answer.

11. Streamlit Dashboard

An interactive Streamlit dashboard combines the project's analytical
components.

The dashboard provides:

Business overview KPIs
Churn distribution
Support-ticket analysis
Customer segmentation
Segment-level metrics
High-value churned customer analysis
Customer explorer
RFM analysis
Raw customer dataset
AI Business Assistant

The AI Business Assistant allows users to ask business questions directly
from the dashboard using the RAG pipeline.

12. Running the Project
    
Create Virtual Environment
python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Configure Groq

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key

Never commit the .env file to GitHub.

13. Run the Data Pipeline

Generate the dataset:

python src/data/generate_data.py

Validate the dataset:

python src/data/data_loader.py

Preprocess:

python src/data/preprocessing.py

Create features:

python src/features/feature_engineering.py


14. Run Machine Learning

Train and compare models:

python src/models/train_model.py

Tune classification threshold:

python src/models/threshold_tuning.py

Save the model:

python src/models/save_model.py

Test prediction:

python -m src.models.predict


15. Run RFM and Segmentation

Create RFM features:

python src/features/rfm_segmentation.py

Run K-Means segmentation:

python src/models/customer_segmentation.py

Profile the segments:

python src/features/segment_profiling.py

16. Run RAG

Build the vector store:

python -m src.rag.vector_store

Run the RAG pipeline:

python -m src.rag.rag_pipeline


17. Run FastAPI

python -m uvicorn app.main:app --reload

Open the Swagger interface:

http://127.0.0.1:8000/docs


18. Run Streamlit
    
streamlit run app/dashboard.py

Open:

http://localhost:8501
Limitations

This project uses synthetic data for demonstration purposes.

Important limitations include:

The dataset does not represent real customer behavior.
The model performance should not be interpreted as production-level
performance.
RFM recency is a tenure-derived proxy because transaction dates are not
available.
Customer segmentation does not directly predict churn.
Model predictions represent probabilities rather than guaranteed
customer outcomes.
Production deployment would require monitoring, model validation,
security, logging and real customer data.
Future Improvements

Possible future improvements include:

Use real transaction timestamps for RFM recency
Hyperparameter optimization
Cross-validation
Model explainability using SHAP
Model monitoring and drift detection
Authentication for APIs
Cloud deployment
Production vector database
Hybrid retrieval
RAG evaluation framework
Automated testing and CI/CD


Key Skills Demonstrated

Python
SQL
Pandas
NumPy
Scikit-learn
XGBoost
Statistical Analysis
Feature Engineering
Logistic Regression
Random Forest
Classification
Class Imbalance
Threshold Tuning
K-Means
RFM Analysis
FastAPI
REST APIs
SQLite
SQLAlchemy
Sentence Transformers
Embeddings
ChromaDB
Groq
LLM Applications
RAG
Streamlit
Git/GitHub

Author

Samseera

AI/ML Engineer | Data Science | Generative AI

Python | SQL | Machine Learning | GenAI | RAG
