import pandas as pd
import seaborn as sns
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import tree

df = pd.read_csv("/content/bank_marketing (1).csv")
df

#define features and target
x = df.iloc[:,:-1]
y = df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# prompt: # Identify categorical features (object type)
# categorical_features = x.select_dtypes(include=['object']).columns
# # Create a LabelEncoder for each categorical feature
# for feature in categorical_features:
#     le = LabelEncoder()
#     x[feature] = le.fit_transform(x[feature])  # Apply Label Encoding
# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
# # Create and train the model
# model = GaussianNB()
# model.fit(X_train, y_train)  # Now the data should be numerical
# # Make predictions
# y_pred = model.predict(X_test)

from sklearn.preprocessing import LabelEncoder

# Identify categorical features (object type)
categorical_features = x.select_dtypes(include=['object']).columns

# Create a LabelEncoder for each categorical feature
for feature in categorical_features:
    le = LabelEncoder()
    x[feature] = le.fit_transform(x[feature])  # Apply Label Encoding

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)  # Now the data should be numerical

# Make predictions
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ... (Your existing code for data loading, splitting, and model training)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='yes')
recall = recall_score(y_test, y_pred, pos_label='yes')
f1 = f1_score(y_test, y_pred, pos_label='yes')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Identify categorical features (object type)
categorical_features = x.select_dtypes(include=['object']).columns

# Create a LabelEncoder for each categorical feature
for feature in categorical_features:
    le = LabelEncoder()
    x[feature] = le.fit_transform(x[feature])  # Apply Label Encoding

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)  # Now the data should be numerical

# Make predictions
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ... (Your existing code for data loading, splitting, and model training)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='yes')
recall = recall_score(y_test, y_pred, pos_label='yes')
f1 = f1_score(y_test, y_pred, pos_label='yes')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))