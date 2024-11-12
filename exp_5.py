import pandas as pd
import seaborn as sns
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import tree

df = pd.read_csv("/content/cake (1).csv")

#define features and target
x = df.iloc[:,:-1]
y = df.iloc[:, -1]

#splitting into ttraining and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

# Create an SVM classifier
svm_model = SVC(kernel="linear")  # You can adjust the kernel and other parameters

from sklearn import svm

decision_model = svm.SVC()

decision_model.fit(x_train,y_train)

y_pred = decision_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion matrix elements
TN, FP, FN, TP = conf_matrix.ravel()

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Assuming you have y_true and y_pred
conf_matrix = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate metrics
precision = precision_score(y_test, y_pred, pos_label='Muffin')  # Update pos_label as per your dataset
recall = recall_score(y_test, y_pred, pos_label='Muffin')
f1 = f1_score(y_test, y_pred, pos_label='Muffin')

# Specificity, False Positive Rate, False Negative Rate
specificity = TN / (TN + FP)
false_positive_rate = FP / (FP + TN)
false_negative_rate = FN / (FN + TP)

# Print results
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("F1-Score:", f1)
print("False Positive Rate:", false_positive_rate)
print("False Negative Rate:", false_negative_rate)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Assuming you have y_true and y_pred
conf_matrix = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate metrics
precision = precision_score(y_test, y_pred, pos_label='Cupcake')
recall = recall_score(y_test, y_pred, pos_label='Cupcake')
f1 = f1_score(y_test, y_pred, pos_label='Cupcake')

# Specificity, False Positive Rate, False Negative Rate
specificity = TN / (TN + FP)
false_positive_rate = FP / (FP + TN)
false_negative_rate = FN / (FN + TP)

# Print results
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("F1-Score:", f1)
print("False Positive Rate:", false_positive_rate)
print("False Negative Rate:", false_negative_rate)

# Create a simple dataset
X, y = make_classification(n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

# Fit the SVM model
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Create a mesh to plot the decision boundary
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the decision boundary
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and scatter plot
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary for Linear SVM')
plt.show()

# Fit the SVM model
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Create a mesh to plot the decision boundary
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the decision boundary
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and scatter plot
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary for Linear SVM (Cupcake vs Muffin)')
plt.show()