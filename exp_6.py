import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('/content/bank_marketing (1).csv') # Replace with your actual file path

# Check column names to find the correct target column
print("deposit:", data.columns)

# Assuming the target column is named 'y' based on common practice and the error message.
# Update this if your target column has a different name.
target_column = 'deposit'

# Step 2: Convert categorical columns to numeric (One-Hot Encoding)
data = pd.get_dummies(data, drop_first=True) # drop_first=True to avoid multicollinearity

# The target column name after one-hot encoding
target_column = 'deposit_yes' # Changed from 'deposit' to 'deposit_yes'
#deposit = 'deposit_yes' # This line is no longer needed
y = data[target_column] # Using the updated target column name

# Step 4: Split data into training and test sets
X = data.drop(columns=[target_column]) # Drop the target column from features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Initialize and train the Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 6: Make predictions on the test set y_pred_rf = rf_classifier.predict(X_test)
y_pred_rf = rf_classifier.predict(X_test)

# Step 7: Model evaluation - Accuracy, Precision, Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, f1_score

# Calculating basic metrics
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)

# Printing basic metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Calculating confusion matrix elements
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()

# Sensitivity (Recall is often used as sensitivity)
sensitivity = tp / (tp + fn)

# Specificity
specificity = tn / (tn + fp)

# False Positives and False Negatives
false_positive = fp
false_negative = fn

# Printing additional metrics
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("False Positive Rate:", false_positive / (false_positive + tn))
print("False Negative Rate:", false_negative / (false_negative + tp))

# Step 8: Confusion Matrix and Heatmap
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"]) # Replace with actual class names if needed plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt. show()

import numpy as np
import matplotlib.pyplot as plt

# Assuming rf_classifier is your trained Random Forest model
importances = rf_classifier.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]  # Sort in descending order

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.title("Feature Importances in Random Forest")
plt.bar(range(len(features)), importances[indices], color="skyblue", align="center")
plt.xticks(range(len(features)), features[indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()