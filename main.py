# main.py
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv(r"D:\MUFG\heart_disease_dataset Capstone project 2.csv")

print("Dataset Preview:")
print(data.head())

# Features & target
X = data.drop("heart_disease", axis=1)
y = data["heart_disease"]

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy", max_depth=6, min_samples_split=4, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=3, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        C=1.5, solver="liblinear", max_iter=2000, random_state=42
    ),
    "SVM": SVC(
        kernel="rbf", C=3, gamma=0.1, probability=True, random_state=42
    ),
}

# Evaluate
results = {}
print("\nModel Performance After Optimization:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print("-------------------------------------------------------------")
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

print("\n-------------------------------------------------------------")
print("Model Comparison Summary:")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# Best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n✅ Best Model: {best_model_name} ({results[best_model_name]:.4f} Accuracy)")

# Save model, scaler, and feature list
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_columns.json", "w") as f:
    json.dump(X.columns.tolist(), f)

print("\n✅ Model, scaler, and feature columns saved successfully!")
