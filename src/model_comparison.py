import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv("../data/simulated_fintech_customers.csv")

# Drop unnecessary columns
df = df.drop(columns=["signup_date", "last_active", "name"])

# Apply one-hot encoding to categorical features
categorical_cols = ["country", "device", "plan"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features and target variable
X = df.drop("churn", axis=1)
y = df["churn"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train and evaluate each model
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

import pandas as pd

# Convert the list of results to a DataFrame
df_results = pd.DataFrame(results)

# Sort by F1 Score in descending order
df_results = df_results.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

# Print the sorted results
print(df_results)

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure directory exists
os.makedirs("../charts/comparison_charts", exist_ok=True)

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(data=df_results, x="F1 Score", y="Model", hue="Model", legend= False, palette="viridis")
plt.title("Model Comparison by F1 Score")
plt.tight_layout()
plt.savefig("../charts/comparison_charts/model_f1_comparison.png")
plt.show()













