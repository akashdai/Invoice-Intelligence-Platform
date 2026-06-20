# train.py

import os
import joblib

from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features
)

from modeling_evaluation import (
    train_random_forest,
    evaluate_classifier
)


# Features Used
FEATURES = [

    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'total_item_quantity',
    'total_item_dollars'

]

TARGET = 'invoice_risk_label'


def main():

    # ------------------------------
    # File Paths
    # ------------------------------

    project_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    db_path = os.path.join(
        project_dir,
        "data",
        "inventory.db"
    )

    model_path = os.path.join(
        project_dir,
        "models",
        "best_invoice_risk_model.pkl"
    )

    # ------------------------------
    # Load Data
    # ------------------------------

    print("Loading Data...")

    df = load_invoice_data(db_path)

    # ------------------------------
    # Prepare Data
    # ------------------------------

    print("Creating Risk Labels...")

    df = apply_labels(df)

    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    # ------------------------------
    # Train Model
    # ------------------------------

    print("Training Random Forest...")

    model = train_random_forest(
        X_train_scaled,
        y_train
    )

    # ------------------------------
    # Evaluate Model
    # ------------------------------

    print("Evaluating Model...")

    evaluate_classifier(
        model,
        X_test_scaled,
        y_test
    )

    # ------------------------------
    # Save Model
    # ------------------------------

    # Create models folder if it does not exist
    # Save Model

    os.makedirs(
    os.path.dirname(model_path),
    exist_ok=True
)

    joblib.dump(
    model,
    model_path
)

    print("\nModel Saved Successfully!")
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    main()


