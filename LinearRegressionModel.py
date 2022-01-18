#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("student.csv", index_col=0)
print(df.head())


# In[3]:


# cheching for null values
print(df.isnull().sum())


# In[4]:

# catagorical values
print(df.select_dtypes(["category", "object"]))


# In[5]:

# numerical values
print(df.select_dtypes(exclude=["category", "object"]))


# In[6]:


# catagorical encoding for the dataset

for colname in df.select_dtypes(["category", "object"]):
    df[colname], _ = df[colname].factorize()


# In[7]:

# co relational matrix
corr_matrix = df.corr()


# In[33]:

# heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(data=corr_matrix, annot=False)
plt.savefig("heatmap.png")


# In[9]:


# from the heat map we can see that G1,G2 have a very high co-relation with G3 the target variable.
# Medu,Mjob,Fedu also are co-retated to each other but have less co relation with the G3 value.
# Simmilarly goout, Dalc,Walc are co-retated to each other but have very less co relation with the G3 value.


# In[10]:

# selecting columns wiht high co-relation with the G3 value
relations = []
for ind, val in corr_matrix['G3'].iteritems():
    if val >= 0.7 or val <= -0.7:
        relations.append(ind)
print(relations)


# In[34]:

# ploting a histogram
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

sns.distplot(df['G1'], kde=True, ax=axes[0])
axes[0].set_title('G1')
sns.distplot(df['G2'], kde=True, ax=axes[1])
axes[1].set_title('G2')
sns.distplot(df['G3'], kde=True, ax=axes[2])
axes[2].set_title('G3')

plt.savefig("distplots.png")


# In[35]:

# plotting scatterplots
fig2, axes2 = plt.subplots(3, 3, figsize=(20, 10), sharey=True)

sns.regplot(df['G1'], df['G3'], ax=axes2[0, 0])
sns.regplot(df['G2'], df['G3'], ax=axes2[0, 1])
sns.regplot((df['G1']+df['G2'])/2, df['G3'], ax=axes2[0, 2])

sns.regplot(df['Medu'], df['G3'], ax=axes2[1, 0])
sns.regplot(df['failures'], df['G3'], ax=axes2[1, 1])
sns.regplot(df['higher'], df['G3'], ax=axes2[1, 2])

sns.scatterplot((df['G1']+df['G2'])/2, df['G3'],
                ax=axes2[2, 0], hue=df['Medu'])
sns.scatterplot((df['G1']+df['G2'])/2, df['G3'],
                ax=axes2[2, 1], hue=df['failures'])
sns.scatterplot((df['G1']+df['G2'])/2, df['G3'],
                ax=axes2[2, 2], hue=df['higher'])

plt.savefig("scatterplots.png")


# In[13]:

# shuffle the dataset
shuffled_df = df.reindex(np.random.permutation(df.index))


# In[25]:

# normalize the values
X = (shuffled_df['G1']+shuffled_df['G2'])/40


# In[26]:


y = shuffled_df['G3']/20


# In[27]:

# Split the dataset into test and train

train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=0)

print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)


# In[28]:

# training the model

model = LinearRegression()

model.fit(np.array(train_X).reshape(-1, 1), np.array(train_y).reshape(-1, 1))


# In[36]:


# In[30]:


print(model.coef_, model.intercept_)


# In[31]:

# validation of the model
pred_y = model.predict(np.array(val_X).reshape(-1, 1))

mse = mean_squared_error(val_y, pred_y)
rmse = sqrt(mse)
r2 = r2_score(val_y, pred_y)

print(mse, rmse, r2)


# In[ ]:
