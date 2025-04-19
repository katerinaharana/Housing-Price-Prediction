#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


import os
print(f"Current folder: {os.getcwd()}")


# In[3]:


cd ../notebooks


# In[4]:


nb_path = os.path.abspath('')
file_path = os.path.join(nb_path, '../data/Housing.csv')

df = pd.read_csv(file_path)
df.head()


# In[5]:


df.shape


# In[6]:


df.nunique()


# In[7]:


df.isna().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


df.info()


# In[10]:


#encoding


# In[11]:


df_object=df.select_dtypes(["object"])


# In[12]:


# display unique values and counts
for col in df_object.columns:
    print(f"\nColumn: {col}")
    print(f"Unique count: {df_object[col].nunique()}")
    print(f"Unique values: {df_object[col].unique()}")


# In[13]:


# map boolean valued columns
bool_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in bool_cols:
    df[col] = df[col].map(lambda x: 1 if x == 'yes' else 0)


# furnished status: do both ordinal and one hot encoding to use them accordingly later depending on the ml model we will use
# Ordinal encoding
ordinal_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
df['furnishingstatus_ordinal'] = df['furnishingstatus'].map(ordinal_map)

# One-Hot encoding
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)




# In[14]:


df


# In[15]:


# Only include columns with few unique values (not area and price)
columns_to_plot = [col for col in df.columns if df[col].nunique() <= 10]

# Generate pie charts for each
for col in columns_to_plot:
    plt.figure()
    df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, labeldistance=1.1,  wedgeprops={'linewidth': 1.8, 'edgecolor': 'white'})
    plt.title(f'Distribution of {col}')
    plt.ylabel('')  # Hide y-axis label
    plt.tight_layout()
    plt.show()


# In[16]:


correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[17]:


from sklearn.ensemble import IsolationForest

# Isolation Forest
features = df.select_dtypes(include=['int64', 'float64'])
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['outlier'] = iso_forest.fit_predict(features)

#number of outliers and preview
num_outliers = (df['outlier'] == -1).sum()
print(f"Number of outliers detected: {num_outliers}")
print(df[df['outlier'] == -1].head())


# In[18]:


from mpl_toolkits.mplot3d import Axes3D

# top 3 features most correlated with price
correlations = df.corr(numeric_only=True)['price'].drop('price')
top_features = correlations.abs().nlargest(3).index.tolist()


inliers = df[df['outlier'] == 1]
outliers = df[df['outlier'] == -1]

# 3D scatter plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    inliers[top_features[0]],
    inliers[top_features[1]],
    inliers[top_features[2]],
    c='lightblue',
    label='Inlier',
    alpha=0.6
)

ax.scatter(
    outliers[top_features[0]],
    outliers[top_features[1]],
    outliers[top_features[2]],
    c='red',
    label='Outlier',
    alpha=0.9
)

ax.set_xlabel(top_features[0])
ax.set_ylabel(top_features[1])
ax.set_zlabel(top_features[2])
ax.set_title("3D Scatter Plot of Inliers and Outliers")
ax.legend()
plt.tight_layout()
plt.show()


# In[19]:


# remove outliers
df = df[df['outlier'] == 1].copy()

# Drop outlier column
df.drop(columns=['outlier'], inplace=True)


# In[20]:


# Feature Engineering: Add  features
df['area_per_bedroom'] = df['area'] / df['bedrooms'].replace(0, 1)
df['area_per_bathroom'] = df['area'] / df['bathrooms'].replace(0, 1)
df['is_fully_equipped'] = (df['airconditioning'] & df['hotwaterheating'] & df['parking']).astype(int)



# In[21]:


print(f"Current folder: {os.getcwd()}")


# In[23]:


get_ipython().system('jupyter nbconvert --to script 01_data_preparation.ipynb')


# In[24]:


file_path = os.path.join(nb_path, '../data/housing_cleaned.csv')

df.to_csv(file_path, index=False)


# In[ ]:





# In[ ]:




