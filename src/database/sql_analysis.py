from sqlalchemy import text

from src.database.database import engine


def run_query(query):
    """Execute a SQL query and return the results."""

    with engine.connect() as connection:
        result = connection.execute(text(query))
        return result.fetchall()


def main():

    print("=" * 60)
    print("SQL CUSTOMER ANALYTICS")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Total customers
    # ---------------------------------------------------------

    query = """
    SELECT COUNT(*) AS total_customers
    FROM customers;
    """

    result = run_query(query)

    print("\n1. TOTAL CUSTOMERS")
    print(f"Total customers: {result[0][0]}")


    # ---------------------------------------------------------
    # 2. Churn distribution
    # ---------------------------------------------------------

    query = """
    SELECT
        churn,
        COUNT(*) AS customer_count
    FROM customers
    GROUP BY churn
    ORDER BY churn;
    """

    result = run_query(query)

    print("\n2. CHURN DISTRIBUTION")

    for row in result:
        print(
            f"Churn: {row[0]} | "
            f"Customers: {row[1]}"
        )


    # ---------------------------------------------------------
    # 3. Overall churn rate
    # ---------------------------------------------------------

    query = """
    SELECT
        ROUND(
            100.0 * SUM(
                CASE
                    WHEN churn = 1 THEN 1
                    ELSE 0
                END
            ) / COUNT(*),
            2
        ) AS churn_rate
    FROM customers;
    """

    result = run_query(query)

    print("\n3. OVERALL CHURN RATE")
    print(f"Churn rate: {result[0][0]}%")


    # ---------------------------------------------------------
    # 4. Average monthly charges by churn status
    # ---------------------------------------------------------

    query = """
    SELECT
        churn,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
    FROM customers
    GROUP BY churn
    ORDER BY churn;
    """

    result = run_query(query)

    print("\n4. AVERAGE MONTHLY CHARGES BY CHURN")

    for row in result:
        print(
            f"Churn: {row[0]} | "
            f"Average Monthly Charges: {row[1]}"
        )


    # ---------------------------------------------------------
    # 5. Average support tickets by churn status
    # ---------------------------------------------------------

    query = """
    SELECT
        churn,
        ROUND(AVG(support_tickets), 2) AS avg_support_tickets
    FROM customers
    GROUP BY churn
    ORDER BY churn;
    """

    result = run_query(query)

    print("\n5. AVERAGE SUPPORT TICKETS BY CHURN")

    for row in result:
        print(
            f"Churn: {row[0]} | "
            f"Average Support Tickets: {row[1]}"
        )


    # ---------------------------------------------------------
    # 6. Customers with high support tickets
    # ---------------------------------------------------------

    query = """
    SELECT
        customer_id,
        support_tickets,
        churn
    FROM customers
    WHERE support_tickets >= 5
    ORDER BY support_tickets DESC
    LIMIT 10;
    """

    result = run_query(query)

    print("\n6. TOP CUSTOMERS WITH HIGH SUPPORT TICKETS")

    for row in result:
        print(
            f"Customer: {row[0]} | "
            f"Tickets: {row[1]} | "
            f"Churn: {row[2]}"
        )


    # ---------------------------------------------------------
    # 7. Customer risk classification using CASE WHEN
    # ---------------------------------------------------------

    query = """
    SELECT
        customer_id,
        support_tickets,
        tenure_months,
        CASE
            WHEN support_tickets >= 5
                 AND tenure_months < 24
                THEN 'HIGH RISK'

            WHEN support_tickets >= 3
                 OR tenure_months < 36
                THEN 'MEDIUM RISK'

            ELSE 'LOW RISK'
        END AS risk_category
    FROM customers
    ORDER BY support_tickets DESC
    LIMIT 10;
    """

    result = run_query(query)

    print("\n7. CUSTOMER RISK CLASSIFICATION")

    for row in result:
        print(
            f"Customer: {row[0]} | "
            f"Tickets: {row[1]} | "
            f"Tenure: {row[2]} | "
            f"Risk: {row[3]}"
        )


    # ---------------------------------------------------------
    # 8. Churn rate by support ticket count
    # GROUP BY + HAVING
    # ---------------------------------------------------------

    query = """
    SELECT
        support_tickets,
        COUNT(*) AS customer_count,
        ROUND(AVG(churn), 3) AS churn_rate
    FROM customers
    GROUP BY support_tickets
    HAVING COUNT(*) >= 20
    ORDER BY churn_rate DESC;
    """

    result = run_query(query)

    print("\n8. CHURN RATE BY SUPPORT TICKET COUNT")

    for row in result:
        print(
            f"Tickets: {row[0]} | "
            f"Customers: {row[1]} | "
            f"Churn Rate: {row[2]:.3f}"
        )


    # ---------------------------------------------------------
    # 9. High-value churned customers
    # Subquery
    # ---------------------------------------------------------

    query = """
    SELECT
        customer_id,
        estimated_lifetime_value,
        churn
    FROM customers
    WHERE churn = 1
      AND estimated_lifetime_value > (
          SELECT AVG(estimated_lifetime_value)
          FROM customers
      )
    ORDER BY estimated_lifetime_value DESC
    LIMIT 10;
    """

    result = run_query(query)

    print("\n9. HIGH-VALUE CHURNED CUSTOMERS")

    for row in result:
        print(
            f"Customer: {row[0]} | "
            f"Lifetime Value: {row[1]:.2f} | "
            f"Churn: {row[2]}"
        )


    # ---------------------------------------------------------
    # 10. High-risk customers using CTE
    # ---------------------------------------------------------

    query = """
    WITH customer_metrics AS (
        SELECT
            customer_id,
            support_tickets,
            tenure_months,
            monthly_charges,
            churn
        FROM customers
    )

    SELECT
        customer_id,
        support_tickets,
        tenure_months,
        monthly_charges,
        churn
    FROM customer_metrics
    WHERE support_tickets >= 4
      AND tenure_months < 36
    ORDER BY support_tickets DESC
    LIMIT 10;
    """

    result = run_query(query)

    print("\n10. HIGH-RISK CUSTOMERS USING CTE")

    for row in result:
        print(
            f"Customer: {row[0]} | "
            f"Tickets: {row[1]} | "
            f"Tenure: {row[2]} | "
            f"Charges: {row[3]:.2f} | "
            f"Churn: {row[4]}"
        )


    # ---------------------------------------------------------
    # 11. Rank customers by lifetime value
    # Window Function
    # ---------------------------------------------------------

    query = """
    SELECT
        customer_id,
        estimated_lifetime_value,
        RANK() OVER (
            ORDER BY estimated_lifetime_value DESC
        ) AS value_rank
    FROM customers
    ORDER BY estimated_lifetime_value DESC
    LIMIT 10;
    """

    result = run_query(query)

    print("\n11. TOP CUSTOMERS BY LIFETIME VALUE")

    for row in result:
        print(
            f"Customer: {row[0]} | "
            f"Lifetime Value: {row[1]:.2f} | "
            f"Rank: {row[2]}"
        )


    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SQL ANALYTICS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()