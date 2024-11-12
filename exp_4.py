import pandas as pd
import seaborn as sns
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn import tree

df = pd.read_csv('/content/titanic.csv')
print(df)

# Drop columns that are not useful
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, columns=['Gender', 'Embarked'], drop_first=True)

#define features and target
x = df.iloc[:,:-1]
y = df.iloc[:, -1]

#splitting into ttraining and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

decision_model = tree.DecisionTreeClassifier()

decision_model.fit(x_train,y_train)

y_pred = decision_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion matrix elements
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate additional metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)  # Sensitivity
specificity = TN / (TN + FP)           # Specificity
f1 = f1_score(y_test, y_pred)
false_positive_rate = FP / (FP + TN)
false_negative_rate = FN / (FN + TP)

# Output results
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Sensitivity (Recall): {recall:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'False Positive Rate: {false_positive_rate:.2f}')
print(f'False Negative Rate: {false_negative_rate:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Plot bar chart for metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'False Positive Rate', 'False Negative Rate']
values = [accuracy, precision, recall, specificity, f1, false_positive_rate, false_negative_rate]

plt.figure(figsize=(8, 5))
plt.barh(metrics, values, color='skyblue')
plt.xlim(0, 1)
plt.title('Model Performance Metrics')
for index, value in enumerate(values):
    plt.text(value + 0.01, index, f'{value:.2f}')
plt.show()

# Step 12: Install required libraries
!apt-get install -y graphviz
!pip install graphviz

# Step 13: Import the necessary modules for visualization
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier

# Assuming X and y are already defined and you want to use DecisionTreeClassifier
model = DecisionTreeClassifier() # Create a DecisionTreeClassifier instance
model.fit(x_train, y_train) # Fit the model to your data (replace X and y with your actual data)


# Step 14: Visualize the decision tree
plt.figure(figsize=(20,10))  # Set the size of the plot
plot_tree(model, filled=True, feature_names=x_train.columns, class_names=["Did not survive", "Survived"])
plt.show()

# Alternatively, you can export the decision tree in Graphviz DOT format
dot_data = export_graphviz(model, out_file=None,
                           feature_names=x_train.columns,
                           class_names=['Did not survive', 'Survived'],
                           filled=True, rounded=True,
                           special_characters=True)

# Create a Graphviz source object and render the tree
graph = graphviz.Source(dot_data)
graph.render("titanic_tree")  # Saves the tree to a file
graph  # Displays the tree inline

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()