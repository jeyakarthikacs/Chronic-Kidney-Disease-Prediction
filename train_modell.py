import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 1. Load dataset
data = pd.read_csv("Kidney_data.csv")

# 2. Clean target column
data["classification"] = (
    data["classification"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"notckd": 0, "ckd": 1})
)

# 3. Map object columns
for col, mapping in [
    ("htn",   {"no": 0, "yes": 1}),
    ("dm",    {"no": 0, "yes": 1}),
    ("appet", {"poor": 0, "good": 1}),
    ("pc",    {"normal": 0, "abnormal": 1}),
]:
    data[col] = (
        data[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(0)
        .astype(int)
    )

# 4. Convert numeric columns
numeric_feats = ["age","sc","sg","hemo","al","rc"]
for col in numeric_feats:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 5. Fill missing values
all_features = ["age","sc","sg","htn","hemo","dm","al","appet","rc","pc"]
data[all_features] = data[all_features].fillna(data[all_features].mean())

# 6. Train/test split
X = data[all_features]
y = data["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Models to compare
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB()
}

# 8. Evaluate models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

    if name == "Random Forest":
        joblib.dump(model, "best_model_random_forest.pkl")

# 9. Create DataFrame of results
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print("\nModel Comparison:\n", results_df)

# 10. Plot the comparison
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Plot F1 Score
sns.barplot(x="Model", y="F1 Score", data=results_df, palette="viridis")
plt.title("F1 Score Comparison of Classifiers")
plt.ylabel("F1 Score")
plt.xlabel("Classifier")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
