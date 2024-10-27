#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,recall_score
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Parkinsson disease.csv')


# In[3]:


print(df.shape)
print(df.info())
df.head()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# 

# In[6]:


df.drop(columns='name',inplace=True)


# In[7]:


df.hist(figsize=(25,20));


# In[8]:


df.skew()


# In[9]:


correl=df.drop(columns='status').corr()
plt.figure(figsize=(20,20))
sns.heatmap(correl,annot=True,cmap='OrRd')
plt.show()


# In[10]:


X = df.drop(columns='status')
y=df['status']


# In[11]:


#Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
X=scaler.fit_transform(X)
y=y


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[13]:


score_baseline = y.value_counts(normalize=True).max()
score_baseline


# In[14]:


lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[15]:


pred_lr=lr.predict(X_test)
# get accuracy score
score_lr = accuracy_score(y_test, pred_lr)
print(f'Accuracy Score : {score_lr}')
print(f'Recall Score : {recall_score(y_test,pred_lr)}')


# In[16]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred_lr)
plt.figure(figsize=(8,6))
fg=sns.heatmap(cm,annot=True,cmap="Reds")
figure=fg.get_figure()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Output Confusion Matrix");


# In[17]:


pd.DataFrame({'actual':y_test,'predict':pred_lr})


# In[18]:


# Get the coefficients of each feature
coef = abs(lr.coef_[0])

# Sort the coefficients in descending order
sorted_idx = np.argsort(coef)[::-1]
sorted_coef = coef[sorted_idx]
sorted_features = df.drop(columns='status').columns[sorted_idx]

# Plot the sorted coefficients
plt.bar(range(len(sorted_coef)), sorted_coef)
plt.xticks(range(len(sorted_coef)), sorted_features, rotation=90)
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.title('Feature Importances for Logistic Regression Model')
plt.show()

