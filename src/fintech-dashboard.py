import os
os.chdir(os.path.dirname(__file__))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create directory if it doesn't exist
os.makedirs("../charts/eda_charts", exist_ok=True)

# Load the dataset
df = pd.read_csv("../data/simulated_fintech_customers.csv")

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], kde=True, bins=20, color='skyblue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("../charts/eda_charts/age_distribution.png")
plt.show()

# Churn Status Count
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', hue='churn', data=df, palette='Set2', legend=False)
plt.title("Churn Status Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.savefig("../charts/eda_charts/churn_status.png")
plt.show()

# Country Distribution
plt.figure(figsize=(8, 5))
sns.countplot(y='country', hue='country', data=df, order=df['country'].value_counts().index, palette='coolwarm', legend=False)
plt.title("Country Distribution")
plt.xlabel("Count")
plt.ylabel("Country")
plt.savefig("../charts/eda_charts/country_distribution.png")
plt.show()

# Device Preference
plt.figure(figsize=(6, 4))
sns.countplot(x='device', hue='device', data=df, palette='pastel', legend=False)
plt.title("Device Preference")
plt.xlabel("Device")
plt.ylabel("Count")
plt.savefig("../charts/eda_charts/device_preference.png")
plt.show()

# Monthly Spending by Plan
plt.figure(figsize=(8, 5))
sns.boxplot(x='plan', y='monthly_spent', hue='plan', data=df, palette='muted', legend=False)
plt.title("Monthly Spending by Plan")
plt.xlabel("Plan Type")
plt.ylabel("Monthly Spent")
plt.savefig("../charts/eda_charts/spending_by_plan.png")
plt.show()

# Plan Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='plan', hue='plan', data=df, palette='Set3', legend=False)
plt.title("Plan Distribution")
plt.xlabel("Plan")
plt.ylabel("Count")
plt.savefig("../charts/eda_charts/plan_distribution.png")
plt.show()

# Spending vs Churn (Scatter)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='monthly_spent', y='age', hue='churn', data=df, palette='cool')
plt.title("Monthly Spending vs Age by Churn Status")
plt.xlabel("Monthly Spent")
plt.ylabel("Age")
plt.legend(title="Churn")
plt.savefig("../charts/eda_charts/spending_vs_churn.png")
plt.show()

# ----- Model Preparation -----
# Feature engineering
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['last_active'] = pd.to_datetime(df['last_active'])
df['active_days'] = (df['last_active'] - df['signup_date']).dt.days

# Drop irrelevant columns
columns_to_drop = [col for col in ['ID', 'name', 'signup_date', 'last_active'] if col in df.columns]
df_model = df.drop(columns=columns_to_drop)

# One-hot encoding for categorical features
df_model = pd.get_dummies(df_model, drop_first=True)

# Display to check the new dataset
print(df_model.head())

# Show first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset info
print("\nData Info:")
print(df.info())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Split the dataset
X = df_model.drop('churn', axis=1)
y = df_model['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# Create directory if not exists
os.makedirs("../charts/model_charts", exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("../charts/model_charts/confusion_matrix.png")
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("../charts/model_charts/roc_curve.png")
plt.show()

# Feature Importance (only if model supports it)
try:
    importances = model.feature_importances_
    features = X.columns
    feat_importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
    feat_importance_df = feat_importance_df.sort_values(by="Importance", ascending=False)

    sns.barplot(x="Importance", y="Feature", data=feat_importance_df, palette="viridis")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("../charts/model_charts/feature_importance.png")
    plt.show()
except AttributeError:
    print("Feature importance is not available for this model.")

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.2f}")



      



























