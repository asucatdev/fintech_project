import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Create output folder if it doesn't exist
os.makedirs("../charts/model_charts_rf", exist_ok=True)

# Load dataset
df = pd.read_csv("../data/simulated_fintech_customers.csv")

# Preprocessing
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['last_active'] = pd.to_datetime(df['last_active'])
df['active_days'] = (df['last_active'] - df['signup_date']).dt.days
df_model = df.drop(['name', 'signup_date', 'last_active'], axis=1)

# One-hot encode categorical variables
df_model = pd.get_dummies(df_model, columns=['country', 'device', 'plan'], drop_first=True)

# Features and target
X = df_model.drop('churn', axis=1)
y = df_model['churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.savefig("../charts/model_charts_rf/confusion_matrix_rf.png")
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='orange')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.savefig("../charts/model_charts_rf/roc_curve_rf.png")
plt.show()

# Feature importance
importances = rf.feature_importances_
feat_names = X.columns
feat_imp = pd.DataFrame({"Feature": feat_names, "Importance": importances})
feat_imp = feat_imp.sort_values("Importance", ascending=False)

sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("../charts/model_charts_rf/feature_importances_rf.png")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.2f}")












