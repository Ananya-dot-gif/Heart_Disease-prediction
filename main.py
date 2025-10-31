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

# ----------------------------------------
# 1️⃣ Load Dataset
# ----------------------------------------
data_path = r"D:\MUFG\heart_disease_dataset Capstone project 2.csv"
data = pd.read_csv(data_path)

print("✅ Dataset Loaded Successfully")
print("Shape:", data.shape)
print(data.head())

# ----------------------------------------
# 2️⃣ Split Features and Target
# ----------------------------------------
if "heart_disease" not in data.columns:
    raise ValueError("❌ 'heart_disease' column not found in dataset!")

X = data.drop("heart_disease", axis=1)
y = data["heart_disease"]

# ----------------------------------------
# 3️⃣ Train-Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------------------
# 4️⃣ Data Standardization
# ----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------
# 5️⃣ Define Models
# ----------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, criterion="entropy", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=42),
    "SVM": SVC(kernel="rbf", C=3, gamma=0.1, probability=True, random_state=42)
}

# ----------------------------------------
# 6️⃣ Train & Evaluate Models
# ----------------------------------------
results = {}
print("\n📊 Model Training and Evaluation:")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n✅ Best Model: {best_model_name} ({results[best_model_name]:.4f} Accuracy)")

# ----------------------------------------
# 7️⃣ Save Model, Scaler, and Features
# ----------------------------------------
# Ensure correct types before saving
if not hasattr(scaler, "transform"):
    raise TypeError("❌ Scaler is not a valid StandardScaler object!")

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_columns.json", "w") as f:
    json.dump(X.columns.tolist(), f)

print("\n🎉 Files saved successfully:")
print("   • best_model.pkl")
print("   • scaler.pkl")
print("   • feature_columns.json")

# ----------------------------------------
# 8️⃣ Test Saved Files (Optional)
# ----------------------------------------
try:
    with open("scaler.pkl", "rb") as f:
        test_scaler = pickle.load(f)
    print("\n🧠 Scaler test passed ✅:", hasattr(test_scaler, "transform"))
except Exception as e:
    print("⚠️ Scaler test failed:", e)
