##1. Heatmap (Using Seaborn)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Sample data for heatmap
data = np.random.rand(10, 10)  # 10x10 matrix with random values

# Plot heatmap
sns.heatmap(data, annot=True, cmap="coolwarm")
plt.show()

##2. Bar Chart (Using Matplotlib)
import matplotlib.pyplot as plt

# Sample data for bar chart
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 2, 5]

# Plot bar chart
plt.bar(categories, values, color='skyblue')
plt.xlabel("Categories")
plt.ylabel("Values")
plt.title("Sample Bar Chart")
plt.show()

##3. Line Plot (Using Matplotlib)
import matplotlib.pyplot as plt

# Sample data for line plot
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Plot line chart
plt.plot(x, y, marker='o', linestyle='-', color='green')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Sample Line Plot")
plt.show()

##4. Scatter Plot (Using Matplotlib)
import matplotlib.pyplot as plt

# Sample data for scatter plot
x = [1, 2, 3, 4, 5]
y = [5, 7, 8, 5, 10]

# Plot scatter plot
plt.scatter(x, y, color='red')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Sample Scatter Plot")
plt.show()
