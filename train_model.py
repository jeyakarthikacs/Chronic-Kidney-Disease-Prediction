import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load
data = pd.read_csv("Kidney_data.csv")

# 2. Map target to 0/1
#   adjust these keys to whatever your CSV actually uses ("ckd"/"notckd", "yes"/"no", etc.)
data["classification"] = (
    data["classification"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"notckd": 0, "ckd": 1})
)

# 3. Explicit, deterministic mappings for the four object columns:
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
        .fillna(0)   # in case of unseen values
        .astype(int)
    )

# 4. Ensure the rest of your features are numeric:
numeric_feats = ["age","sc","sg","hemo","al","rc"]
for col in numeric_feats:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 5. Fill missing values **only** in numeric columns
all_features = ["age","sc","sg","htn","hemo","dm","al","appet","rc","pc"]
data[all_features] = data[all_features].fillna(data[all_features].mean())

# 6. Split & train
X = data[all_features]
y = data["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Save
joblib.dump(clf, "xbg_model.pkl")
print("Model saved as xbg_model.pkl")
