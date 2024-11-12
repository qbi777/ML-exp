import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import Markdown as md

#reading data's
df_departmentdata = pd.read_csv("/content/department_data.csv")
df_employeeDetailsData = pd.read_csv("/content/employee_details_data.csv")
df_employeeData = pd.read_csv("/content/employee_data.csv")

#printing the data
print(df_departmentdata)
print(df_employeeDetailsData)
print(df_employeeData)

#shape
print(df_departmentdata.shape)
print(df_employeeDetailsData.shape)
print(df_employeeData.shape)
#head()
#printing the data
print(df_departmentdata.head())
print(df_employeeDetailsData.head())
print(df_employeeData.head())

"""employee_id"""

df_employeeData['employee_id'].describe()

df_employeeData[df_employeeData['employee_id']<=0]

df_employeeData[df_employeeData['employee_id']<=0].shape

df_employeeData.drop(df_employeeData[df_employeeData['employee_id'] <= 0].index, inplace=True)

df_employeeData.shape

#average monthly hours
df_employeeData['avg_monthly_hrs'].describe()

df_departmentdata['dept_id'].unique()

df_employeeData['department'].unique()

#replaceing
df_employeeData['department'].replace({'-IT':'D00-IT'},inplace = True)

df_employeeData['department'].unique()

#joing the datasets

df_empData = pd.merge(df_employeeData,df_employeeDetailsData, how = 'left', on = 'employee_id')

print(df_empData.head())
print(df_empData.shape)

df = pd.merge(df_empData,df_departmentdata,how = 'left', left_on='department',right_on='dept_id')

df = df.drop('department', axis=1)

#converting a dataframe to csv with csv file name
df.to_csv('Employee_original_data.csv')

df.info()

df.describe(include ='all')

df.isnull().sum()[df.isnull().sum() != 0]

missing = df.isnull().sum()[df.isnull().sum() != 0]
missing = pd.DataFrame(missing.reset_index())
missing.rename(columns={'index': 'features', 0: 'missing_count'}, inplace=True)
missing['missing_count_percentage'] = ((missing['missing_count'])/df.shape[0])*100

plt.figure()
sns.barplot(y = missing['features'], x= missing['missing_count_percentage'])

df.drop(['filed_complaint', 'recently_promoted'], axis=1, inplace=True)

df.dropna(subset = ['dept_id','dept_name','dept_head'], inplace= True)

#impute missing values
plt.plot(figsize=(15,10))
sns.boxplot(df['last_evaluation'])

#last evaluation with no out layers
#fillna to fill the values of na naan
df['last_evaluation'].fillna(df['last_evaluation'].mean(), inplace=True)

sns.boxplot(df['satisfaction'])

