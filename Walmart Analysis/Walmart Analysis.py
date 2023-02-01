#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('train.csv')


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


train.shape


# In[6]:


train.isnull().sum()


# In[7]:


test = pd.read_csv('test.csv')
test.head()


# In[8]:


test.info()


# In[9]:


test.shape


# In[10]:


test.isnull().sum()


# In[11]:


features = pd.read_csv('features.csv')
features.head()


# In[12]:


features.info()


# In[13]:


features.shape


# In[14]:


features.isnull().sum()


# In[15]:


stores = pd.read_csv('stores.csv')
stores.head()


# In[16]:


stores.info()


# In[17]:


stores.shape


# In[18]:


stores.isnull().sum()


# In[19]:


df = features.merge(stores, how='inner', on='Store')


# **merge()**
# 
# Parameters :
# - right : DataFrame or named Series
# - how : {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’
# - on : label or list
# - left_on : label or list, or array-like
# - right_on : label or list, or array-like
# - left_index : bool, default False
# - right_index : bool, default False
# - sort : bool, default False
# - suffixes : tuple of (str, str), default (‘_x’, ‘_y’)
# - copy : bool, default True
# - indicator : bool or str, default False
# - validate : str, optional
# 
# Returns : A DataFrame of the two merged objects.

# In[20]:


df.head()


# In[21]:


df.info()


# In[22]:


import datetime
df['Date'] = pd.to_datetime(df['Date'])


# When a csv file is imported and a Data Frame is made, the Date time objects in the file are read as a string object rather a Date Time object and hence it's very tough to perform operations like _Time Difference_ on a string rather a _Date Time object_.
# 
# **Pandas to_datetime() method** helps to convert string Date time into Python Date time object.
# 
# **self.df["date"] = pd.to_datetime(self.df["date"]).dt.date returns just the DATE as OBJECT datatype.**

# In[23]:


train['Date'] = pd.to_datetime(train['Date'])


# In[24]:


final = train.merge(df, on=['Store', 'Date', 'IsHoliday'])


# In[25]:


final.head()


# In[26]:


final.info()


# In[27]:


final['Week'] = final['Date'].dt.isocalendar().week


# ![image.png](attachment:image.png)

# In[28]:


final['Year'] = final['Date'].dt.isocalendar().year


# In[29]:


final['Date'].dt.day


# In[30]:


final.head()


# In[31]:


final.tail()


# ### Scatter Plot of weekly sales & column (we pass as argument)

# In[32]:


def scatter(df, col):
    plt.figure(figsize = (15, 12))
    plt.scatter(x = df['Weekly_Sales'], y=df[col])
    plt.title('Weekly_Sales vs ' + str(col))
    plt.xlabel('Weekly_Sales')
    plt.ylabel(col)


# In[33]:


scatter(final, 'Store')


# In[34]:


scatter(final, 'Dept')


# In[35]:


scatter(final, 'IsHoliday')


# ### Average weekly sales in 2011?

# In[36]:


weekly_sales_2011 = final[final['Year'] == 2011].groupby(['Week'])['Weekly_Sales'].mean()


# In[37]:


weekly_sales_2012 = final[final['Year'] == 2012].groupby(['Week'])['Weekly_Sales'].mean()


# In[38]:


weekly_sales_2010 = final[final['Year'] == 2010].groupby(['Week'])['Weekly_Sales'].mean()


# In[39]:


sns.lineplot(x=weekly_sales_2010.index, y=weekly_sales_2010.values, color='red')


# In[40]:


sns.lineplot(x = weekly_sales_2011.index, y = weekly_sales_2011.values, color = 'blue')


# In[41]:


sns.lineplot(x = weekly_sales_2012.index, y = weekly_sales_2012.values, color = 'green')


# In[42]:


plt.figure(figsize = (20, 15))
sns.lineplot(x = weekly_sales_2010.index, y = weekly_sales_2010.values, color = 'red')
sns.lineplot(x = weekly_sales_2011.index, y = weekly_sales_2011.values, color = 'blue')
sns.lineplot(x = weekly_sales_2012.index, y = weekly_sales_2012.values, color = 'green')
plt.xticks(np.arange(1, 55, step=1))
plt.grid()
plt.title('Average Weekly Sales per Year')
plt.xlabel('Week')
plt.ylabel('Sales')
plt.legend(['2010', '2011', '2012'])


# In[43]:


sns.histplot(final['Weekly_Sales'])


# In[44]:


sns.boxplot(x = 'Type', y = 'Size', data = final)


# ### Store with highest average weekly sales?

# In[45]:


store_weekly_sales = final['Weekly_Sales'].groupby(final['Store']).mean()


# In[46]:


store_weekly_sales


# In[47]:


store_weekly_sales_df = pd.DataFrame(store_weekly_sales)


# In[48]:


store_weekly_sales_df


# In[49]:


store_weekly_sales_df.sort_values("Weekly_Sales", ascending=False).style.bar(color = '#FFD200')


# - Store 20 - Highest sales
# - Store 5  - Lowest sales

# In[50]:


final[final['Store'] == 20]['Type'].unique()


# In[51]:


final[final['Store'] == 5]['Type'].unique()


# In[52]:


plt.figure(figsize = (25,15))
sns.barplot(x = store_weekly_sales_df.index, y = store_weekly_sales.values)
plt.grid()
plt.title('Average weekly sales per store', fontsize = 30)
plt.xlabel('Store', fontsize = 20)
plt.ylabel('Weekly Sales', fontsize = 20)


# ### Average Weekly Sales per Department?

# In[53]:


dept_weekly_sales = final.groupby(['Dept'])['Weekly_Sales'].mean()


# In[54]:


dept_weekly_sales


# In[55]:


dept_weekly_sales_df = pd.DataFrame(dept_weekly_sales)


# In[56]:


dept_weekly_sales_df


# In[57]:


dept_weekly_sales_df.sort_values('Weekly_Sales', ascending = False).style.bar()


# - Department 92 - Highest Sales
# - Department 43 - Lowest Sales

# In[58]:


plt.figure(figsize=(25, 15))
sns.barplot(x = dept_weekly_sales.index, y = dept_weekly_sales.values)
plt.grid()
plt.title('Average Weekly Sales per Department', fontsize = 30)
plt.xlabel('Department', fontsize = 20)
plt.ylabel('Weekly Sales', fontsize = 20)


# ### Correlation b/w features

# In[59]:


plt.figure(figsize = (15, 8))
sns.heatmap(final.corr(), annot = True)


# # Conclusions:
# 
# - Department 92 having highest sales
# - Store 20 having highest sales
# - Store 5 have lowest sales
# - Department 43 have lowest sales
# - All years display same pattern and we can clearly see higher sales in the end weeks of the year.
# - We can see some higher sales scenarios when isHoliday is true
# - Type a stores have the larger size whereas type c stores have the smaller size.
# - Store 20 that have highest sales belongs so type a whereas store 5 that have lowest sales belong to type c
# - Higher correlation between fuel prices and years.

# ### Authored by:
# 
# [Soumya Kushwaha](https://github.com/Soumya-Kushwaha)
