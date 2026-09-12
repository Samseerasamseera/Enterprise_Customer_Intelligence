import sys
import os

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from src.rag.rag_pipeline import ask_question


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise Customer Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Enterprise Customer Intelligence")

st.markdown(
    "Customer Churn Prediction, Segmentation, RFM Analysis "
    "and AI-powered Business Intelligence"
)


# ============================================================
# LOAD DATA
# ============================================================

customer_path = (
    "data/processed/customer_features.csv"
)

segment_path = (
    "data/processed/customer_segments_profiled.csv"
)

customer_df = pd.read_csv(
    customer_path
)

segment_df = pd.read_csv(
    segment_path
)


# ============================================================
# CREATE RFM FEATURES
# ============================================================

# Recency proxy because the synthetic dataset does not
# contain actual transaction dates.

customer_df["recency"] = (
    customer_df["tenure_months"].max()
    - customer_df["tenure_months"]
)

# Frequency
customer_df["frequency"] = (
    customer_df["total_purchases"]
)

# Monetary
customer_df["monetary"] = (
    customer_df["total_purchases"]
    * customer_df["avg_transaction_value"]
)


# ============================================================
# MERGE SEGMENT INFORMATION
# ============================================================

segment_columns = [
    "customer_id"
]

if "segment" in segment_df.columns:

    segment_columns.append(
        "segment"
    )

if "segment_name" in segment_df.columns:

    segment_columns.append(
        "segment_name"
    )

segment_data = segment_df[
    segment_columns
].copy()


df = customer_df.merge(
    segment_data,
    on="customer_id",
    how="left"
)


# ============================================================
# BUSINESS OVERVIEW
# ============================================================

st.subheader(
    "Business Overview"
)

total_customers = len(df)

churned_customers = int(
    df["churn"].sum()
)

churn_rate = (
    df["churn"].mean()
    * 100
)

avg_support_tickets = (
    df["support_tickets"].mean()
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


with col2:

    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )


with col3:

    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )


with col4:

    st.metric(
        "Avg Support Tickets",
        f"{avg_support_tickets:.2f}"
    )


st.divider()


# ============================================================
# CHURN DISTRIBUTION
# ============================================================

st.subheader(
    "Churn Distribution"
)

churn_counts = (
    df["churn"]
    .value_counts()
    .reset_index()
)

churn_counts.columns = [
    "churn",
    "customers"
]

churn_counts["status"] = (
    churn_counts["churn"]
    .map(
        {
            0: "Not Churned",
            1: "Churned"
        }
    )
)


fig_churn = px.pie(
    churn_counts,
    names="status",
    values="customers",
    title="Customer Churn Distribution"
)


st.plotly_chart(
    fig_churn,
    use_container_width=True
)


# ============================================================
# SUPPORT TICKETS VS CHURN
# ============================================================

st.subheader(
    "Support Tickets vs Churn"
)

support_churn = (
    df.groupby("churn")[
        "support_tickets"
    ]
    .mean()
    .reset_index()
)

support_churn["status"] = (
    support_churn["churn"]
    .map(
        {
            0: "Not Churned",
            1: "Churned"
        }
    )
)


fig_support = px.bar(
    support_churn,
    x="status",
    y="support_tickets",
    title="Average Support Tickets by Churn Status",
    labels={
        "status": "Customer Status",
        "support_tickets": "Average Support Tickets"
    }
)


st.plotly_chart(
    fig_support,
    use_container_width=True
)


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

st.subheader(
    "Customer Segmentation"
)


if "segment_name" in df.columns:

    segment_summary = (
        df.groupby("segment_name")
        .agg(
            customers=(
                "customer_id",
                "count"
            ),
            avg_frequency=(
                "frequency",
                "mean"
            ),
            avg_monetary=(
                "monetary",
                "mean"
            ),
            churn_rate=(
                "churn",
                "mean"
            )
        )
        .reset_index()
    )


    segment_summary[
        "churn_rate"
    ] = (
        segment_summary[
            "churn_rate"
        ] * 100
    )


    st.dataframe(
        segment_summary,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CUSTOMER COUNT BY SEGMENT
    # --------------------------------------------------------

    fig_segment_count = px.bar(
        segment_summary,
        x="segment_name",
        y="customers",
        title="Customers by Segment",
        labels={
            "segment_name": "Customer Segment",
            "customers": "Number of Customers"
        }
    )


    st.plotly_chart(
        fig_segment_count,
        use_container_width=True
    )


    # --------------------------------------------------------
    # MONETARY VALUE BY SEGMENT
    # --------------------------------------------------------

    fig_segment_value = px.bar(
        segment_summary,
        x="segment_name",
        y="avg_monetary",
        title="Average Monetary Value by Segment",
        labels={
            "segment_name": "Customer Segment",
            "avg_monetary": "Average Monetary Value"
        }
    )


    st.plotly_chart(
        fig_segment_value,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CHURN RATE BY SEGMENT
    # --------------------------------------------------------

    fig_segment_churn = px.bar(
        segment_summary,
        x="segment_name",
        y="churn_rate",
        title="Churn Rate by Customer Segment",
        labels={
            "segment_name": "Customer Segment",
            "churn_rate": "Churn Rate (%)"
        }
    )


    st.plotly_chart(
        fig_segment_churn,
        use_container_width=True
    )


else:

    st.warning(
        "Segment information is not available."
    )


# ============================================================
# HIGH-VALUE CHURNED CUSTOMERS
# ============================================================

st.subheader(
    "High-Value Churned Customers"
)


high_value_churned = (
    df[
        df["churn"] == 1
    ]
    .sort_values(
        "estimated_lifetime_value",
        ascending=False
    )
    [
        [
            "customer_id",
            "age",
            "tenure_months",
            "monthly_charges",
            "estimated_lifetime_value",
            "support_tickets",
            "churn"
        ]
    ]
    .head(10)
)


st.dataframe(
    high_value_churned,
    use_container_width=True
)


# ============================================================
# CUSTOMER EXPLORER
# ============================================================

st.subheader(
    "Customer Explorer"
)


customer_ids = (
    df["customer_id"]
    .astype(str)
    .tolist()
)


selected_customer = st.selectbox(
    "Select Customer ID",
    customer_ids
)


selected_id = int(
    selected_customer
)


customer = (
    df[
        df["customer_id"]
        == selected_id
    ]
    .iloc[0]
)


col1, col2, col3 = st.columns(3)


with col1:

    st.write(
        "### Customer Information"
    )

    st.write(
        f"**Customer ID:** "
        f"{int(customer['customer_id'])}"
    )

    st.write(
        f"**Age:** "
        f"{customer['age']:.0f}"
    )

    st.write(
        f"**Tenure:** "
        f"{customer['tenure_months']:.0f} months"
    )


with col2:

    st.write(
        "### Purchase Information"
    )

    st.write(
        f"**Total Purchases:** "
        f"{customer['total_purchases']:.0f}"
    )

    st.write(
        f"**Average Transaction Value:** "
        f"{customer['avg_transaction_value']:.2f}"
    )

    st.write(
        f"**Monetary Value:** "
        f"{customer['monetary']:.2f}"
    )


with col3:

    st.write(
        "### Customer Risk"
    )

    st.write(
        f"**Support Tickets:** "
        f"{customer['support_tickets']:.0f}"
    )


    if customer["churn"] == 1:

        st.error(
            "Customer Churned"
        )

    else:

        st.success(
            "Customer Retained"
        )


    if "segment_name" in df.columns:

        st.write(
            f"**Segment:** "
            f"{customer['segment_name']}"
        )


# ============================================================
# RFM ANALYSIS
# ============================================================

st.subheader(
    "RFM Analysis"
)


rfm_display = df[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "churn"
    ]
].copy()


rfm_display = (
    rfm_display
    .sort_values(
        "monetary",
        ascending=False
    )
    .head(10)
)


st.dataframe(
    rfm_display,
    use_container_width=True
)


# ============================================================
# AI BUSINESS ASSISTANT - RAG
# ============================================================

st.divider()

st.subheader(
    "🤖 AI Business Assistant"
)


st.markdown(
    """
Ask questions about:

- Customer retention
- Customer support
- Customer segmentation
- RFM analysis

The AI assistant retrieves relevant information from the
business knowledge base and generates a grounded response
using the Groq LLM.
"""
)


question = st.text_input(
    "Ask a business question:",
    placeholder=(
        "Example: What should we do for a high risk customer?"
    )
)


if st.button(
    "Ask AI"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching knowledge base and generating answer..."
        ):

            try:

                answer, sources = ask_question(
                    question,
                    top_k=3
                )


                # ------------------------------------------------
                # ANSWER
                # ------------------------------------------------

                st.markdown(
                    "### 💡 Answer"
                )

                st.write(
                    answer
                )


                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                st.markdown(
                    "### 📚 Sources"
                )


                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    st.write(
                        f"{i}. "
                        f"{source['source']} "
                        f"(chunk {source['chunk_id']})"
                    )


            except Exception as e:

                st.error(
                    f"Unable to generate answer: {e}"
                )


# ============================================================
# RAW DATA
# ============================================================

st.subheader(
    "Raw Customer Dataset"
)


with st.expander(
    "View Dataset"
):

    st.dataframe(
        df,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Enterprise Customer Intelligence | "
    "Python • SQL • Machine Learning • RFM • "
    "Customer Segmentation • GenAI • RAG"
)