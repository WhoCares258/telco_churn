import os
import google.generativeai as genai
from xgboost import XGBClassifier
import shap
import pandas as pd

# --- Load model ---
clf = XGBClassifier()
clf.load_model("xgb_churn_model.json")

# --- Load test data ---
X_test = pd.read_csv("X_test.csv")

# Pick one sample
sample = X_test.iloc[[5]]
prediction = clf.predict(sample)
proba = clf.predict_proba(sample)[0, 1]

# --- SHAP explainer ---
explainer = shap.TreeExplainer(clf, feature_perturbation="interventional")
shap_values = explainer.shap_values(sample, check_additivity=False)

shap_df = pd.DataFrame(shap_values, columns=X_test.columns, index=sample.index)
top_features = shap_df.T.sort_values(by=sample.index[0], key=abs, ascending=False).head(5)

# --- Setup Gemini safely ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# --- Interpret model output ourselves ---
label = "Churn" if prediction[0] == 1 else "No Churn"
churn_risk = f"low risk ({proba:.0%})" if proba < 0.5 else f"high risk ({proba:.0%})"

# Pick top positive and negative factors
sorted_features = shap_df.T.sort_values(by=sample.index[0], ascending=False)
positive_factors = sorted_features.head(3).index.tolist()  # push toward churn
negative_factors = sorted_features.tail(3).index.tolist()  # push away from churn

# Build clear summary to feed Gemini
summary = f"""
Prediction: {label} with probability {proba:.2f} → this means {churn_risk} of churn.

Top factors increasing churn risk: {', '.join(positive_factors)}.
Top factors decreasing churn risk: {', '.join(negative_factors)}.

Give a simple, 3-sentence explanation for a business user.
"""
response = model.generate_content(summary)

print("🔮 Prediction:", "Churn" if prediction[0] == 1 else "No Churn", f"(prob={proba:.2f})")
print("\n📊 Top SHAP feature impacts:")
print(top_features)
print("\n🤖 Gemini Explanation:")
print(response.text)
