# modeling_evaluation.py

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def train_random_forest(X_train, y_train):

    # Best Parameters obtained from GridSearchCV

    model = RandomForestClassifier(

        n_estimators=200,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,

        random_state=42
    )

    print("\nTraining Random Forest with Best Parameters")
    print("-------------------------------------------")
    print("n_estimators      :", 200)
    print("max_depth         :", None)
    print("min_samples_split :", 3)
    print("min_samples_leaf  :", 1)

    model.fit(X_train, y_train)

    return model


def evaluate_classifier(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print("\nModel Performance")
    print("----------------------")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return f1