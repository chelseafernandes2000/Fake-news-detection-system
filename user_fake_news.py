#!/usr/bin/env python
# coding: utf-8

# In[89]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


# In[90]:


df=pd.read_csv('train.csv')


# In[91]:


df.dropna(inplace=True)


# In[98]:


title=input('Enter title:')
author=input('Enter author:')



# In[99]:


X_train=df['author']
y_train=df['label']

df_test=pd.read_csv('test.csv')

df_test.head()
df_test.dropna(inplace=True)

df_test=df_test.append({'title' : title , 'author' : author,'text':'ll'} , ignore_index=True)
X_test=df_test['author']


# In[100]:


tf=TfidfVectorizer(stop_words='english')
tf_train=tf.fit_transform(X_train)
tf_test=tf.transform(X_test)
#for keys,value in tf.vocabulary_.items():
    #print(keys,value)
    #if keys=='jessica':
        #tat=value


# In[101]:


linear_clf = PassiveAggressiveClassifier(max_iter=50)
linear_clf.fit(tf_train, y_train)
test_proba=linear_clf._predict_proba_lr(tf_test)[:,1]*100


# In[102]:


res=pd.DataFrame(X_test)
k=pd.DataFrame(df_test['title'])
res['confidence'] = test_proba
res['class']=linear_clf.predict(tf_test)
r=df_test['title']
res['title']=k


# In[103]:


nres=res.to_numpy()
for x in nres:
    if x[3] == title:
        if x[2]==0:
            x[1]=100-x[1]
            print('Reliable news')
        else:
            print('unreliable news')
        print(F"Title: {title} ,Class: {x[2]}, Confidence: {x[1]}")


# In[ ]:





# In[ ]:




