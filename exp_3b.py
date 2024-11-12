import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates, andrews_curves, radviz, scatter_matrix
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy import stats

from IPython.display import set_matplotlib_formats
matplotlib.style.use('ggplot')
import os
from warnings import filterwarnings

filterwarnings('ignore')

# %matplotlib inline
sns.set_context('notebook')
plt.close('all')

data = pd.read_csv('/content/bank_marketing.csv')
data

print(data.describe(include = 'all'))

data.info()

data.isnull().sum()

data.describe()

"""**Filling missing values or dropping rows/columns with missing data**"""

data = data.dropna()
data.fillna(method='ffill', inplace=True)

"""**Encode Categorical Values:** Logistic regression need only numeriacal values"""

data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])

"""**Convert target variables to binary**"""

data['deposit'] = data['deposit'].apply(lambda x: 1 if x == 'yes' else 0)

"""**Split data set:** Split the data into features(x) and target(y)"""

from sklearn.model_selection import train_test_split

X = data.drop(['deposit', 'Unnamed: 0'], axis=1)  # Drop irrelevant columns
y = data['deposit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""# **Train the model**

**Import the models**
"""

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)



model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
print('\nshape of logistic coefficients\n')
print(np.shape(model.coef_))

"""# **Evaluate the model**

**Make prediction**
"""

y_pred = model.predict(X_test)
print(y_pred)

"""**Evaluate performance**"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)

y_pred_scaled = model.predict(X_test_scaled)

"""# **Testing with new data **"""

def predict_subscription(age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome):

    new_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'day': [day],
        'month': [month],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    new_data = pd.get_dummies(new_data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])

    new_data = new_data.reindex(columns=X.columns, fill_value=0)

    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled)

    return "Yes" if prediction[0] == 1 else "No"

age = 30
job = 'services'
marital = 'single'
education = 'secondary'
default = 'no'
balance = 150
housing = 'yes'
loan = 'no'
contact = 'cellular'
day = 15
month = 'may'
duration = 200
campaign = 1
pdays = -1
previous = 0
poutcome = 'unknown'


predicted_subscription = predict_subscription(age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome)
print(f"Predicted Subscription: {predicted_subscription}")

