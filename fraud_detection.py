"""Credit Card Fraud Detection (By Tejasvi)
Desc: Detect fraudulent credit card transactions using simple ML models.
Models used: Logistic Regression, Random Forest, XGBoost """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def load_and_preprocess(filepath: str):
    #Load dataset and scale Time + Amount features.
    df = pd.read_csv("C:\\Users\\shriy\\Desktop\\data\\FraudAnalysis\\creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale only 'Time' and 'Amount'
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))
    X["Time"] = scaler.fit_transform(X["Time"].values.reshape(-1, 1))

    return X, y


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    #Train model and print results.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {model_name} Results ===")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def main():
    print("🔹 Loading dataset...")
    X, y = load_and_preprocess("creditcard.csv")

    print("🔹 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    train_and_evaluate(lr, X_train, X_test, y_train, y_test, "Logistic Regression")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest")

    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    train_and_evaluate(xgb_clf, X_train, X_test, y_train, y_test, "XGBoost")


if __name__ == "__main__":
    main()
