#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymysql
from sqlalchemy import create_engine
import pandas as pd
import getpass 
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
password = getpass.getpass()
connection_string = 'mysql+pymysql://root:' + password + '@localhost/sakila'
engine = create_engine(connection_string)


# In[2]:


query='''select
r.rental_date, 
f.rental_duration,
SUBSTRING(r.rental_date, 1, 7) AS last_month, 
CASE WHEN SUBSTRING(r.rental_date, 1, 7) = '2006-02' THEN 1 ELSE 0 END AS Boolean_check,
f.length, 
f.rating,
f.film_id,
c.name


from rental r
left join inventory i
on r.inventory_id = i.inventory_id

left join film f
on i.film_id = f.film_id

left join film_category fc
on fc.film_id = f.film_id

left join category c
on c.category_id = fc.category_id

order by r.rental_date desc
;'''


# In[3]:


data = pd.read_sql_query(query, engine)
data.head()


# In[ ]:





# In[4]:


data.isna().sum()


# In[5]:


data.describe()


# In[6]:


data = data.set_index('film_id')
data


# In[7]:


data.dtypes


# In[8]:


corr_matrix=data.corr()
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(corr_matrix, annot=True)
plt.show()


# In[9]:


for col in data.select_dtypes(np.number):
    sns.displot(data[col])
    plt.show()


# In[ ]:





# In[10]:


X = data.select_dtypes(include = np.number)

transformer = StandardScaler().fit(X)
x_normalized = transformer.transform(X)
data_num = pd.DataFrame(x_normalized)
data_num.columns = X.columns
data_num.head()


# In[11]:


for col in data.select_dtypes('object'):
    print(data[col].value_counts(), '\n')


# In[12]:


data_cat = data.select_dtypes(include = object)


# In[13]:


data_cat


# In[15]:


data_cat = data_cat.drop(['last_month'], axis =1)


# In[16]:


data_cat= pd.get_dummies(data_cat)
data_cat


# In[17]:


y = data['Boolean_check'] 


# In[18]:


y


# In[ ]:





# In[19]:


data_num


# In[20]:


data_cat


# In[21]:


X=data_cat.join(data_num)


# In[22]:


X


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[24]:


classification = LogisticRegression(random_state=42, max_iter=500) 
classification.fit(X_train, y_train)


# In[25]:


predictions = classification.predict(X_test)


# In[26]:


pd.Series(predictions).value_counts()


# In[28]:


y_test.value_counts()


# In[29]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True,fmt='g')


# In[30]:



from sklearn.metrics import accuracy_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, predictions)))


# In[31]:


from sklearn import metrics
import matplotlib.pyplot as plt

y_pred_proba = classification.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr)


# In[ ]:




