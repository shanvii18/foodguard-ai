import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def train_model():
    df = pd.read_csv("dataset.csv")

    # Encode features
    le_product   = LabelEncoder()
    le_category  = LabelEncoder()
    le_method    = LabelEncoder()
    le_severity  = LabelEncoder()
    le_target    = LabelEncoder()  # health_risk = target

    df["product_enc"]  = le_product.fit_transform(df["product_name"])
    df["category_enc"] = le_category.fit_transform(df["category"])
    df["method_enc"]   = le_method.fit_transform(df["detection_method"])
    df["severity_enc"] = le_severity.fit_transform(df["severity"])
    df["target"]       = le_target.fit_transform(df["health_risk"])

    X = df[["product_enc", "category_enc", "method_enc", "severity_enc"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {acc * 100:.2f}%")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("encoders.pkl", "wb") as f:
        pickle.dump({
            "product":  le_product,
            "category": le_category,
            "method":   le_method,
            "severity": le_severity,
            "target":   le_target,
        }, f)

    print("Model saved!")
    return model

if __name__ == "__main__":
    train_model()
