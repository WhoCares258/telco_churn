from XGBoost import XGBClassifier
import shap

model.save_model('xgb_churn_model.json')

clf = XGBClassifier()
clf.load_model('xgb_churn_model.json')

prediction = clf.predict(X_test)

explainer = shap.TreeExplainer(clf)

features = [
 'SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines',
 'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
 'StreamingMovies','MonthlyCharges','TotalCharges',
 'InternetService_Fiber optic','InternetService_No',
 'Contract_One year','Contract_Two year',
 'PaymentMethod_Credit card (automatic)'
]

