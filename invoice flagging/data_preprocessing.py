# data_preprocessing.py

import sqlite3
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_invoice_data(db_path):

    conn = sqlite3.connect(db_path)

    query = """
    WITH purchase_agg AS (

        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(
                JULIANDAY(p.ReceivingDate) -
                JULIANDAY(p.PODate)
            ) AS avg_receiving_delay

        FROM purchases p

        GROUP BY p.PONumber
    )

    SELECT

        vi.PONumber,

        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,

        ROUND(
            JULIANDAY(vi.InvoiceDate) -
            JULIANDAY(vi.PODate), 0
        ) AS days_po_to_invoice,

        ROUND(
            JULIANDAY(vi.PayDate) -
            JULIANDAY(vi.InvoiceDate), 0
        ) AS days_to_pay,

        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay

    FROM vendor_invoice vi

    LEFT JOIN purchase_agg pa
    ON vi.PONumber = pa.PONumber
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


def create_invoice_risk_label(row):

    amount_mismatch = (
        abs(
            row['invoice_dollars'] -
            row['total_item_dollars']
        ) > 0.01
    )

    high_receiving_delay = (
        row['avg_receiving_delay'] > 30
    )

    if amount_mismatch or high_receiving_delay:
        return 1

    return 0


def apply_labels(df):

    df['invoice_risk_label'] = df.apply(
        create_invoice_risk_label,
        axis=1
    )

    return df


def split_data(df, features, target):

    X = df[features]

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler