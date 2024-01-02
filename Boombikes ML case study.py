#!/usr/bin/env python
# coding: utf-8

# In[378]:


#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# In[379]:


df = pd.read_csv('day.csv')


# ## Getting an overview of the dataset.

# In[380]:


df.head()


# In[381]:


df.info()


# In[382]:


df.shape


# In[383]:


df.describe()


# In[384]:


df.size


# In[385]:


#Before dropping date, let's introduce a days_old variable which indicates how old the business is.

df['days_old'] = (pd.to_datetime(df['dteday'],format= '%d-%m-%Y') - pd.to_datetime('01-01-2018',format= '%d-%m-%Y')).dt.days


# # Data Cleaning

# In[386]:


#Dropping 'instant','casual','registered' and 'dteday'

df = df.drop(columns=['instant','casual','registered','dteday'])


# #### These variables were dropped since instant is the just the serial number of the records and casual + registered = cnt

# In[387]:


df.head()


# In[388]:


df.season.value_counts()


# In[389]:


#Print null counts by column

df.isnull().sum()


# #### As we an see, there are no null values present in our data.

# In[390]:


#Checking number of unique values in each column

df.nunique()


# In[391]:


# Box plots for indepent variables with continuous values

columns = ['temp', 'atemp', 'hum', 'windspeed']
plt.figure(figsize=(18,4))

i = 1
for col in columns:
    plt.subplot(1,4,i)
    sns.boxplot(y=col, data=df)
    i+=1


# #### As we can see from the above plots, there are no outliers to be handled.

# In[392]:


df.corr()


# In[393]:


# Convert season and  weathersit to categorical types

df.season.replace({1:"spring", 2:"summer", 3:"fall", 4:"winter"},inplace = True)

df.weathersit.replace({1:'good',2:'moderate',3:'bad',4:'severe'},inplace = True)

df.mnth = df.mnth.replace({1: 'jan',2: 'feb',3: 'mar',4: 'apr',5: 'may',6: 'jun',
                  7: 'jul',8: 'aug',9: 'sept',10: 'oct',11: 'nov',12: 'dec'})

df.weekday = df.weekday.replace({0: 'sun',1: 'mon',2: 'tue',3: 'wed',4: 'thu',5: 'fri',6: 'sat'})
df.head()


# In[394]:


#Draw pairplots for continuous numeric variables

plt.figure(figsize = (20,30))
sns.pairplot(data=df,vars=['cnt', 'temp', 'atemp', 'hum','windspeed'])
plt.show()


# In[395]:


plt.figure(figsize=(10,15))
sns.pairplot(df)
plt.show()


# In[396]:


# Checking continuous variables relationship with each other

sns.heatmap(df[['temp','atemp','hum','windspeed','cnt']].corr(), cmap='Blues', annot = True)
plt.show()


# In[397]:


#Correlations between numeric variables

cor=df.corr()
sns.heatmap(cor, cmap="BuPu", annot = True)
plt.show()


# In[398]:


# Boxplot for categorical variables to see demands

vars_cat = ['season','yr','mnth','holiday','weekday','workingday','weathersit']
plt.figure(figsize=(15, 15))
for i in enumerate(vars_cat):
    plt.subplot(3,3,i[0]+1)
    sns.boxplot(data=df, x=i[1], y='cnt')
plt.show()


# In[399]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Temp",fontsize=16)
sns.regplot(data=df,y="cnt",x="temp")
plt.xlabel("Temperature")
plt.show()


# ####  Observations:
# 
# 1. Demand for bikes is positively correlated to temp.
# 2. We can observe that 'cnt' is linearly increasing with 'temp' showing linear relation.

# In[400]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Hum",fontsize=16)
sns.regplot(data=df,y="cnt",x="hum")
plt.xlabel("Humidity")
plt.show()


# #### Observations:
# 
# 1. Hum is values are more scattered around.
# 2. Although we can see cnt decreasing with increase in humidity.

# In[401]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Windspeed",fontsize=16)
sns.regplot(data=df,y="cnt",x="windspeed")
plt.show()


# #### Observations:
# 
# 1. Windspeed is values are more scattered around.
# 2. Although we can see cnt decreasing with increase in windspeed.

# In[402]:


num_features = ["temp","atemp","hum","windspeed","cnt"]
plt.figure(figsize=(15,8),dpi=130)
plt.title("Correlation of numeric features",fontsize=16)
sns.heatmap(df[num_features].corr(),annot= True,cmap="mako")
plt.show()


# ####  Observations:
# 
# 1. Temp and Atemp are highly correlated, we can remove one of them, but for now, lets keep them for further analysis.
# 2. Temp and Atemp also have high correlation with cnt.

# In[403]:


df.describe()


# #  Preparing  data for Linear Regression.

# In[404]:


# Creaing dummy variables for all the categorical variables.

df = pd.get_dummies(data=df,columns=["season","mnth","weekday"],drop_first=True)
df = pd.get_dummies(data=df,columns=["weathersit"])


# In[405]:


#Print columns after creating dummies

df.columns


# In[406]:


df.head()


# In[407]:


#X will be all the remaining variables and also our independent variables
X=df

#Train Test split with 70:30 ratio
train, test = train_test_split(df,train_size=0.7, test_size=0.3, random_state=100)


# In[408]:


scaler = MinMaxScaler()


# In[409]:


# Apply scaler() to all the columns except the 'dummy' variables.
#num_vars = ['hum','windspeed','temp','atemp']
num_vars = ['cnt','hum','windspeed','temp','atemp']

train[num_vars] = scaler.fit_transform(train[num_vars])


# In[410]:


train.head()


# In[411]:


y_train = train.pop('cnt')
X_train = train


# # Building the Model

# Fit a regression line through the training data using statsmodels. In statsmodels, you need to explicitly fit a constant using sm.add_constant(X) because if we don't perform this step, statsmodels fits a regression line passing through the origin, by default.
# 
# 

# In[412]:


# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 13)             # running RFE
rfe = rfe.fit(X_train, y_train)


# In[413]:


#Columns selected by RFE and their weights
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# ### Model 1

# #Function to get VIF
# def get_vif():
#     vif = pd.DataFrame()
#     X = X_train_new
#     vif['Features'] = X.columns
#     vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#     vif['VIF'] = round(vif['VIF'], 2)
#     vif = vif.sort_values(by = "VIF", ascending = False)
#     print(vif)

# In[414]:


#Print Columns selected by RFE. We will start with these columns for manual elimination
col = X_train.columns[rfe.support_]
col


# In[415]:


X_train.columns[~rfe.support_]


# In[416]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[417]:


# Adding a constant variable 
 
X_train_rfe = sm.add_constant(X_train_rfe)


# In[418]:


# Create a first fitted model
lm = sm.OLS(y_train,X_train_rfe).fit()


# In[419]:


# Check the parameters obtained

lm.params


# In[420]:


# Print a summary of the linear regression model obtained

print(lm.summary())


# In[421]:


# Calculate the VIFs for the model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[422]:


#const has very high VIF so we can drop it
X_train_new = X_train_rfe.drop(["const"], axis = 1)


# ### Model 2

# In[423]:


# Adding a constant variable 
X_train_lm = sm.add_constant(X_train_new)


# In[424]:


lm = sm.OLS(y_train,X_train_lm).fit()


# In[425]:


print(lm.summary())


# In[426]:


# Calculate the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[427]:


#weathersit_good has very high VIF so we can drop it
X_train_new = X_train_rfe.drop(["weathersit_good"], axis = 1)


# ### Model 3

# In[428]:


# Adding a constant variable 
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()


# In[429]:


lm.summary()


# In[430]:


# Calculate the VIFs for the new model again
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[431]:


X_train_new = X_train_new.drop(['season_spring'], axis=1)


# ###  Model 4

# In[432]:


# Adding a constant variable 
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()


# In[433]:


lm.summary()


# ##  Residual Analysis of train data

# In[434]:


y_train_cnt = lm.predict(X_train_lm)


# In[435]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# ##  Making Predictions using Final model

# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the final model

# In[436]:


#Applying the scaling on the test sets
num_vars = ['cnt','hum','windspeed','temp','atemp']

test[num_vars] = scaler.transform(test[num_vars])


# In[437]:


test.describe()


# In[438]:


#Dividing into X_test and y_test
y_test = test.pop('cnt')
X_test = test


# In[439]:


# Adding constant variable to test dataframe
X_test = sm.add_constant(X_test)


# In[440]:


# predicting using values used by the final model
test_col = X_train_lm.columns
X_test=X_test[test_col[1:]]
# Adding constant variable to test dataframe
X_test = sm.add_constant(X_test)

X_test.info()


# In[441]:


# Making predictions using the fourth model

y_pred = lm.predict(X_test)


# In[442]:


r2_score(y_test, y_pred)


# In[443]:


from sklearn.metrics import mean_squared_error


# In[444]:


mse = mean_squared_error(y_test, y_pred)
mse


# ## Model Evaluation

# #### Let's now plot the graph for actual versus predicted values.
# 

# In[445]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)      


# In[447]:


param = pd.DataFrame(lm.params)
param.insert(0,'Variables',param.index)
param.rename(columns = {0:'Coefficient value'},inplace = True)
param['index'] = list(range(0,12))
param.set_index('index',inplace = True)
param.sort_values(by = 'Coefficient value',ascending = False,inplace = True)
param

