from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evalutation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)

import pandas as pd
import joblib


def main():

    # Load Data
    df = load_vendor_invoice_data(
        db_path="data/inventory.db",
        table_name="vendor_invoice"
    )

    # Prepare Data
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Train Models
    lr_model = train_linear_regression(
        X_train,
        y_train
    )

    dt_model = train_decision_tree(
        X_train,
        y_train
    )

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    # Evaluate Models
    lr_metrics = evaluate_model(
        lr_model,
        X_test,
        y_test
    )

    dt_metrics = evaluate_model(
        dt_model,
        X_test,
        y_test
    )

    rf_metrics = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    results = pd.DataFrame([
        ["Linear Regression", *lr_metrics.values()],
        ["Decision Tree", *dt_metrics.values()],
        ["Random Forest", *rf_metrics.values()]
    ],
    columns=["Model", "MAE", "RMSE", "R2"])

    print("\nModel Comparison")
    print(results)

    # Select Best Model (Lowest MAE)

    best_model_name = results.loc[
        results["MAE"].idxmin(),
        "Model"
    ]

    if best_model_name == "Linear Regression":
        best_model = lr_model

    elif best_model_name == "Decision Tree":
        best_model = dt_model

    else:
        best_model = rf_model

    print(f"\nBest Model: {best_model_name}")

    # Save Best Model

    joblib.dump(
        best_model,
        "best_freight_model.pkl"
    )

    print("Best model saved successfully.")


if __name__ == "__main__":
    main()

