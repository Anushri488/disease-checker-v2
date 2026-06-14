"""
Disease Prediction Model — trained on a real symptom-disease dataset
derived from the public "Disease Prediction Using Machine Learning"
dataset (132 symptoms x 41 diseases).

NOTE ON THE DATA: The source CSV has 4,920 rows, but only ~300 are
unique symptom combinations (5-10 patterns per disease, each repeated
~17-24 times). This is a known characteristic of this public dataset.
We therefore:
  1) Train the production model on the full (duplicated) dataset, as
     is standard practice for this dataset.
  2) Separately report cross-validation accuracy on the DEDUPLICATED
     set (304 unique symptom patterns) as a more honest estimate of
     how well the model distinguishes between diseases, since the
     duplicated version trivially inflates CV/test accuracy.

- Random Forest (compared against Decision Tree and Naive Bayes)
- 41 diseases, 132 symptoms
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

print("Loading real disease-symptom dataset...")

df = pd.read_csv("disease_dataset_clean.csv")
dedup = df.drop_duplicates().reset_index(drop=True)

all_symptoms = [c for c in df.columns if c != "prognosis"]
print(f"Symptoms: {len(all_symptoms)}, Diseases: {df['prognosis'].nunique()}")
print(f"Full dataset: {len(df)} rows | Unique symptom patterns: {len(dedup)}")

X = df[all_symptoms].values
le = LabelEncoder()
y = le.fit_transform(df["prognosis"])

X_dedup = dedup[all_symptoms].values
y_dedup = le.transform(dedup["prognosis"])

# Honest evaluation: 5-fold CV on the deduplicated, unique-pattern dataset
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Cross-validation on DEDUPLICATED data (304 unique patterns) ---")
for name, model in [
    ("Random Forest", RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=42, n_jobs=-1)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Naive Bayes", GaussianNB()),
]:
    try:
        scores = cross_val_score(model, X_dedup, y_dedup, cv=cv, n_jobs=-1)
        print(f"{name}: {scores.mean():.2%} (+/- {scores.std():.2%})")
    except ValueError as e:
        print(f"{name}: could not run CV ({e})")

# --- Train final Random Forest on the FULL dataset (standard for this dataset) ---
print("\n--- Training final model on full dataset ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest test accuracy (full data, duplicated): {rf_acc:.2%}")

print("\nSaving Random Forest as the production model...")
joblib.dump(rf, "model.pkl")
joblib.dump(le, "encoder.pkl")
joblib.dump(all_symptoms, "symptoms_list.pkl")
print("Saved model.pkl, encoder.pkl, symptoms_list.pkl")
