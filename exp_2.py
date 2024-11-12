import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings('ignore')

iris = load_iris()
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target,iris.target_names)

df.head()

df.shape

df.info()

missing_values = df.isnull().sum()
print("\n missing value: n", missing_values)

sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of Iris Datasets", y=1.02)
plt.show()

#df['species_encoded'] = df['species'].cat.codes

df['species_encoded'] = df['species'].astype('category').cat.codes

correlation_matrix = df.drop(columns=['species']).corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', cbar = True)

sns.histplot(data=sns.load_dataset("iris"), x="sepal_length", hue="species", kde=True, palette="viridis"); plt.show()

#plt.figure(figsize=(12, 8))
#sns.boxplot(data=df, orient="h", palette="set2")
#plt.title("Boxplot of Iris Features")
#plt.show()

ax = sns.boxplot(data=iris, orient="h", palette="Set2")

sns.violinplot(x='species', y='sepal_length', data=iris, palette="muted")
plt.title("Violin plot of Sepal Length by Species")
plt.show()

sns.violinplot(x='species', y='sepal_width', data=iris, palette="muted")
plt.title("Violin plot of Sepal Width by Species")
plt.show()

sns.violinplot(x='species', y='petal_length', data=iris, palette="muted")
plt.title("Violin plot of Petal Length by Species")
plt.show()

sns.violinplot(x='species', y='petal_width', data=iris, palette="muted")
plt.title("Violin plot of Petal Width by Species")
plt.show()