# main.py
# -----------------------------
# Heart Disease Detection - Model Training Script (Final Version)
# -----------------------------

import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = pd.read_csv("heart_disease_dataset.csv")  # keep file in same folder
print("✅ Dataset Loaded Successfully!")
print("Shape:", data.shape)
print("Columns:", list(data.columns))

# Separate features and target variable
X = data.drop("heart_disease", axis=1)
y = data["heart_disease"]

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Step 3: Feature Scaling (retain feature names)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# -----------------------------
# Step 4: Model Initialization
# -----------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_split=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=3, random_state=42),
    "Logistic Regression": LogisticRegression(C=1.5, solver="liblinear", max_iter=2000, random_state=42),
    "SVM": SVC(kernel="rbf", C=3, gamma=0.1, probability=True, random_state=42),
}

# -----------------------------
# Step 5: Train & Evaluate Models
# -----------------------------
print("\n🚀 Training Models...\n")
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print("-------------------------------------------------------------")
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# -----------------------------
# Step 6: Select & Save Best Model
# -----------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n🏆 Best Model:", best_model_name)
print(f"✅ Accuracy: {results[best_model_name]:.4f}")

# Save model, scaler, and feature columns
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)

print("\n✅ Files Saved: model.pkl, scaler.pkl, feature_columns.json")
