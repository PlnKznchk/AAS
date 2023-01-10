#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports 


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.stats.api as sms
from statsmodels.compat import lzip


# In[2]:


#Read the data

data = pd.read_csv('analysis-8.csv', sep=",")


# In[3]:


#Check the data

data.head()


# In[4]:


#Describe the data

data.describe()


# In[5]:


# Fit the model

model2 = smf.ols("ACI_Total ~ GDP_per_capita + Unemployment_total + WHI + Literacy_total + Life_expectancy + Depression_rate", data).fit()
print(model2.summary())


# In[6]:


model2 = smf.ols("ACI_Total ~ GDP_per_capita + Unemployment_total + WHI + Literacy_total + Life_expectancy + Depression_rate", data).fit(cov_type='HC3')
print(model2.summary())


# In[7]:


# Check the assumptions

#Residuals are normally distributed

stats.probplot(model2.resid, dist="norm", plot= plt)
plt.title("Multiple Model Residuals Q-Q Plot")


# In[8]:


# Check the assumptions

#No Multicollinearity 'VIF'


y, X = dmatrices('ACI_Total ~  GDP_per_capita + Unemployment_total + WHI + Literacy_total + Life_expectancy + Depression_rate', data=data, return_type='dataframe')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns

vif


# In[9]:


data = pd.read_csv('analysis-7.csv', sep=",")

data.corr()


# In[10]:


corr = data.corr()
sns.heatmap(corr, cmap = 'twilight', annot= True);


# In[11]:


# Check the assumptions

#Durbin-Watson test to validate the Independence of Errors assumption

round(sm.stats.stattools.durbin_watson(model2.resid),2)


# In[12]:


# Check the assumptions

#Breusch-Pagan test to validate the Homoscedasticity assumption

name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
test = sms.het_breuschpagan(model2.resid, model2.model.exog)
test_round = [round(item, 2) for item in test]
lzip(name, test_round)


# In[13]:


name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(model2.resid)
test_round = [round(item, 2) for item in test]
lzip(name, test_round)


# In[14]:


# Check the assumptions

#Linearity

sns.lmplot(x='GDP_per_capita',y='ACI_Total',data = data,aspect=2,height=6)
plt.xlabel('GDP per capita')
plt.ylabel('Alcohol Consumption Index')
plt.title('Income Level Vs Alcohol Consumption');

plt.savefig('myimage.png', format='png', dpi=1200)


# In[15]:


sns.lmplot(x='Unemployment_total',y='ACI_Total',data = data,aspect=2,height=6)
plt.xlabel('Unemployment rate')
plt.ylabel('Alcohol Consumption Index')
plt.title('Unemployment Rate Vs Alcohol Consumption');


# In[16]:


sns.lmplot(x='WHI',y='ACI_Total',data = data,aspect=2,height=6)
plt.xlabel('World Hapiness Index')
plt.ylabel('Alcohol Consumption Index')
plt.title('Happiness level Vs Alcohol Consumption');


# In[17]:


sns.lmplot(x='Literacy_total',y='ACI_Total',data = data,aspect=2,height=6)
plt.xlabel('Literacy rate')
plt.ylabel('Alcohol Consumption Index')
plt.title('Literacy rate Vs Alcohol Consumption');


# In[18]:


sns.lmplot(x='Life_expectancy',y='ACI_Total',data = data,aspect=2,height=6)
plt.xlabel('Life Expectancy')
plt.ylabel('Alcohol Consumption Index')
plt.title('Life Expectancy Vs Alcohol Consumption');


# In[19]:


sns.lmplot(x='Depression_rate',y='ACI_Total',data = data,aspect=2,height=6)
plt.xlabel('Depression rate')
plt.ylabel('Alcohol Consumption Index')
plt.title('Depression Rate Vs Alcohol Consumption');


# In[20]:


# Distribution of data

f= plt.figure(figsize=(15,4))

ax=f.add_subplot(121)
sns.distplot(data['ACI_Total'],bins=50,color='r',ax=ax)
ax.set_title('Distribution of Alcohol Consumption')


# In[21]:


f= plt.figure(figsize=(15,4))

ax=f.add_subplot(121)
sns.distplot(data['GDP_per_capita'],bins=50,color='b',ax=ax)
ax.set_title('Distribution of Income')


# In[22]:


f= plt.figure(figsize=(15,4))

ax=f.add_subplot(121)
sns.distplot(data['Unemployment_total'],bins=50,color='g',ax=ax)
ax.set_title('Distribution of Unemplyment level')


# In[23]:


f= plt.figure(figsize=(15,4))

ax=f.add_subplot(121)
sns.distplot(data['WHI'],bins=50,color='y',ax=ax)
ax.set_title('Distribution of Happiness level')


# In[24]:


f= plt.figure(figsize=(15,4))

ax=f.add_subplot(121)
sns.distplot(data['Literacy_total'],bins=50,color='c',ax=ax)
ax.set_title('Distribution of Literacy level')


# In[25]:


f= plt.figure(figsize=(15,4))

ax=f.add_subplot(121)
sns.distplot(data['Life_expectancy'],bins=50,color='k',ax=ax)
ax.set_title('Distribution of Life expactancy')


# In[26]:


f= plt.figure(figsize=(15,4))

ax=f.add_subplot(121)
sns.distplot(data['Depression_rate'],bins=50,color='m',ax=ax)
ax.set_title('Distribution of Depression level')

