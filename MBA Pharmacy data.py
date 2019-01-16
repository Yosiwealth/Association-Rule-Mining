#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
data = pd.read_csv('Pharmacy Data.csv')


# In[7]:


data.head()


# columns = ['Date', 'Detail', 'Price', 'Stock', 'Staff', 'Qty', 'Qty/Pk']
# data = data.drop(columns, axis=1)

# foo = lambda a: ", ".join(a)
# df = data.groupby(by = 'Ref').agg({'Article':foo})#.reset_index()

# In[8]:


basket = (data.groupby(['Ref','Article'])['Qty/Pk'].sum().unstack().reset_index().fillna(0).set_index('Ref'))


# In[9]:


basket.head()


# In[10]:


def encoder(x):
    if x <= 0:
        return 0
    else:
        return 1


# In[11]:


basket = basket.applymap(encoder)


# In[ ]:





# In[12]:


basket.head()


# In[13]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[14]:


items_together = apriori(basket, min_support = 0.001, use_colnames=True)


# In[13]:


items_together


# In[15]:


rules = association_rules(items_together, metric='lift', min_threshold=1)


# In[16]:


rules


# In[ ]:




