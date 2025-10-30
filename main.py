import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("heart_disease_dataset.csv")

# Split features and labels
X = data.drop("heart_disease", axis=1)
y = data["heart_disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_split=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=3, random_state=42),
    "Logistic Regression": LogisticRegression(C=1.5, solver="liblinear", max_iter=2000, random_state=42),
    "SVM": SVC(kernel="rbf", C=3, gamma=0.1, probability=True, random_state=42)
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

# Save trained model, scaler, and features
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
json.dump(list(X.columns), open("feature_columns.json", "w"))

print(f"✅ Best model saved: {best_model_name}")
