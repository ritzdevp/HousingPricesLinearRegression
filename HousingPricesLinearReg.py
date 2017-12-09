
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


# In[4]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[6]:


boston


# In[9]:


df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)


# In[10]:


df_x.describe()


# In[11]:


reg = linear_model.LinearRegression()


# In[12]:


#now split data into training and testing data


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4)


# In[14]:


reg.fit(x_train, y_train)


# In[16]:


reg.coef_


# In[17]:


a = reg.predict(x_test)


# In[23]:


a[4]


# In[21]:


y_test


# In[24]:


#mean square error


# In[25]:


# 'a' is our predictions


# In[26]:


np.mean((a-y_test)**2)

