#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing basic libraries

import numpy as np
import pandas as pd


# In[2]:


#reading the dataset

data=pd.read_csv('Advertising.csv')


# In[3]:


data.head()


# In[4]:


# processing the data

data.shape


# In[5]:


data.info()


# In[6]:


#we don't need the 1st column so let's drop that

data=data.iloc[:,1:]


# In[7]:


data.tail()


# In[8]:


#check for null values

data.isna().sum()


# In[9]:


data.describe()


# In[10]:


#Data Visulaization

import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


sns.pairplot(data,kind="reg");


# In[12]:


fig,axs= plt.subplots(1,3,sharey=True)
data.plot(kind="scatter",x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind="scatter",x='Radio',y='Sales',ax=axs[1],figsize=(16,8))
data.plot(kind="scatter",x='Newspaper',y='Sales',ax=axs[2],figsize=(16,8))


# In[13]:


#rmoving the outlier from newspaper

data=data[data['Newspaper']<=90]
data.shape


# In[14]:


data.corr()


# In[15]:


# Separating input and output data

x=data.drop(columns=['Sales'])
y=data['Sales']


# In[16]:


x.head()


# In[17]:


y.head()


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.tail()


# In[19]:


y_test.tail()


# # Model building & deployment

# In[20]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[21]:


column_trans=make_column_transformer((OneHotEncoder(sparse=False),[]),remainder='passthrough')
scaler=StandardScaler()
oe=OrdinalEncoder()


# In[22]:


#Linear Regression Model

from sklearn.linear_model import LinearRegression
lr= LinearRegression()
pipe=make_pipeline(column_trans,scaler,lr)
pipe.fit(x_train,y_train)
y_pred_lr=pipe.predict(x_test)
r2_score(y_test,y_pred_lr)


# In[23]:


#Decision Tree Regression Model

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=0)
pipe=make_pipeline(column_trans,scaler,dt)
pipe.fit(x_train,y_train)
y_pred_dt=pipe.predict(x_test)
r2_score(y_test,y_pred_dt)


# In[24]:


#Random Forest Regression Model

from sklearn.ensemble import RandomForestRegressor
r=RandomForestRegressor(n_estimators=10,random_state=0)
pipe=make_pipeline(column_trans,scaler,r)
pipe.fit(x_train,y_train)
y_pred_r=pipe.predict(x_test)
r2_score(y_test,y_pred_r)


# In[ ]:





# In[26]:


#Let's Check predict function working Good or Not

pipe.predict([[283.6,42.0,66.2]]) #Original ans 25.5


# In[27]:


pipe.predict([[112.9,17.4,38.6]]) #Original ans 11.9


# # The Model For Deployment
# 

# In[28]:


import pickle
pickle.dump(pipe,open('sales.pkl','wb'))


# In[ ]:





# In[ ]:




