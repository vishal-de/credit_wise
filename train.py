"""
Train script for CreditWise loan approval model.
Replicates the pipeline from credit_wise.ipynb and saves everything
needed to serve predictions in the Streamlit app.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# ---------- 1. Load data ----------
df = pd.read_csv("loan_approval_data.csv")

# ---------- 2. Handle missing values ----------
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

# ---------- 3. Drop ID ----------
df = df.drop("Applicant_ID", axis=1)

# ---------- 4. Encoding ----------
le_education = LabelEncoder()
df["Education_Level"] = le_education.fit_transform(df["Education_Level"])

le_target = LabelEncoder()
df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])

ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[ohe_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

# ---------- 5. Feature engineering (final version from notebook) ----------
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y = df["Loan_Approved"]

# ---------- 6. Train/test split + scaling ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 7. Train Naive Bayes (best model per notebook) ----------
model = GaussianNB()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("Naive Bayes Model")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- 8. Save everything the app needs ----------
joblib.dump({
    "model": model,
    "scaler": scaler,
    "ohe": ohe,
    "le_education": le_education,
    "le_target": le_target,
    "ohe_cols": ohe_cols,
    "feature_columns": list(X.columns),   # exact column order model expects
}, "credit_model.pkl")

print("\nSaved credit_model.pkl")
