# main.py
# -----------------------------
# Heart Disease Detection - Model Training Script
# -----------------------------
# Run this file once locally to train and save the model and scaler

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = pd.read_csv("heart_disease_dataset.csv")  # Make sure file is in same folder

# Separate features and target
X = data.drop("heart_disease", axis=1)
y = data["heart_disease"]

print("✅ Dataset Loaded Successfully!")
print(f"Shape: {X.shape}")
print(f"Columns: {list(X.columns)}")

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Step 3: Feature Scaling (retain DataFrame structure)
# -----------------------------
scaler = StandardScaler()
scaler.fit(X_train)

# Keep feature names in scaled data (important for deployment)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# -----------------------------
# Step 4: Train Models
# -----------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_split=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=4, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel="rbf", C=1, probability=True)
}

print("\n🚀 Training Models...\n")
accuracy_scores = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    accuracy_scores[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# -----------------------------
# Step 5: Choose Best Model
# -----------------------------
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name} with Accuracy = {accuracy_scores[best_model_name]:.4f}")

# -----------------------------
# Step 6: Save Model and Scaler
# -----------------------------
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model and Scaler saved successfully!")
print("Files generated: model.pkl, scaler.pkl")
