#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('winequality-red.csv')


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['fixed acidity'],kde=False,bins=50)


# In[8]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['volatile acidity'],kde=False,bins=50)


# In[9]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['citric acid'],kde=False,bins=50)


# In[10]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['free sulfur dioxide'],kde=False,bins=50)


# In[11]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['total sulfur dioxide'],kde=False,bins=50)


# In[12]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['sulphates'],kde=False,bins=50)


# In[13]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['density'],kde=False,bins=50)


# In[14]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['alcohol'],kde=False,bins=50)


# In[15]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['pH'],kde=False,bins=50)


# In[16]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['quality'],kde=False,bins=50)


# In[17]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['chlorides'],kde=False,bins=50)


# In[18]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['residual sugar'],kde=False,bins=50)


# In[19]:


sns.boxplot(x='quality',y='citric acid',data=data)


# In[20]:


sns.boxplot(x='quality',y='fixed acidity',data=data)


# In[21]:


sns.boxplot(x='quality',y='volatile acidity',data=data)


# In[22]:


sns.boxplot(x='quality',y='residual sugar',data=data)


# In[23]:


sns.boxplot(x='quality',y='chlorides',data=data)


# In[24]:


sns.boxplot(x='quality',y='free sulfur dioxide',data=data)


# In[25]:


sns.boxplot(x='quality',y='total sulfur dioxide',data=data)


# In[26]:


sns.boxplot(x='quality',y='density',data=data)


# In[27]:


sns.boxplot(x='quality',y='alcohol',data=data)


# In[28]:


sns.boxplot(x='quality',y='pH',data=data)


# In[29]:


sns.boxplot(x='quality',y='sulphates',data=data)


# In[30]:


sns.scatterplot(x='quality',y='fixed acidity',data=data)


# In[31]:


sns.scatterplot(x='quality',y='volatile acidity',data=data)


# In[32]:


sns.scatterplot(x='quality',y='citric acid',data=data)


# In[33]:


sns.scatterplot(x='quality',y='alcohol',data=data)


# In[34]:


sns.scatterplot(x='quality',y='sulphates',data=data)


# In[35]:


data['quality']=pd.cut(data['quality'],bins=(2,6.5,8),labels=['bad','good'])


# In[36]:


le=LabelEncoder()
new_data=le.fit_transform(data['quality'])


# In[37]:


data['quality']=new_data
data.head()


# In[38]:


X=data.drop('quality',axis=1)
y=data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[39]:


sc = StandardScaler()


# In[40]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[41]:


error_rate = []
for i in range(1,40):
 
 knc = KNeighborsClassifier(n_neighbors=i)
 knc.fit(X_train,y_train)
 pred_i = knc.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))


# In[42]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
 markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[43]:


knc = KNeighborsClassifier(n_neighbors=4)
knc.fit(X_train,y_train)
knc_prediction = knc.predict(X_test)


# In[44]:


print(confusion_matrix(y_test,knc_prediction))


# In[45]:


print(classification_report(y_test,knc_prediction))


# In[46]:


print("Accuracy:{}".format(accuracy_score(y_test,knc_prediction)))


# In[ ]:




