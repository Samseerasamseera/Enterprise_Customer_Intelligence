import pandas as pd
from sqlalchemy import create_engine


# Database location
DATABASE_URL = "sqlite:///data/customer_intelligence.db"

# Create database engine
engine = create_engine(DATABASE_URL)


def load_customer_data():
    """
    Load processed customer data into SQLite database.
    """

    file_path = "data/processed/customer_features.csv"

    df = pd.read_csv(file_path)

    # Store data in SQLite
    df.to_sql(
        "customers",
        con=engine,
        if_exists="replace",
        index=False
    )

    print("=" * 60)
    print("DATABASE LOADING")
    print("=" * 60)
    print(f"Records loaded: {len(df)}")
    print("Table created: customers")
    print("Database: data/customer_intelligence.db")
    print("=" * 60)


if __name__ == "__main__":
    load_customer_data()