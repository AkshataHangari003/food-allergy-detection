import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/food_allergy_dataset1.csv")
df.fillna(0, inplace=True)

df["contains_allergens"] = df["contains_allergens"].astype(str).str.lower()
df["contains_allergens"] = df["contains_allergens"].apply(
    lambda x: 1 if x in ["1", "yes", "true", "present", "contains"] else 0
)

y = df["contains_allergens"]

X = df.drop(["contains_allergens", "food_name"], axis=1, errors="ignore")
X = pd.get_dummies(X)

QUESTIONNAIRE_FEATURES = [
    "early_symptom", "repeat_reaction", "family_history",
    "medicine_taken", "partial_relief",
    "other_food_reaction", "doctor_consulted"
]

for f in QUESTIONNAIRE_FEATURES:
    X[f] = 0

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)

with open("model/allergy_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model trained with questionnaire features")
