
# coding: utf-8

# Importing packages

# In[3]:


import numpy as np
import pandas as pd


# Importing data set

# In[202]:


bayes_home = pd.read_csv("G:/Practise/Bayes Homes Prices/Bayes home prices.csv")


# In[22]:


bayes_home.head(3)


# In[6]:


bayes_home.shape


# In[7]:


bayes_home.size


# In[153]:


bayes_home.describe()


# In[14]:


bayes_home.neighborhood.value_counts().head()


# Selecting features

# In[166]:


bayes_home.iloc[ : , 4:15].head()


# Creating a new data set with relevant features

# In[203]:


df = bayes_home.iloc[ : , 4:15]
df = pd.concat([df, bayes_home['zindexvalue']], axis = 1)
df.shape


# In[167]:


df.head()


# In[204]:


df.zindexvalue = df.zindexvalue.str.replace(',','')


# Convert the zindexvalue into a numeric column

# In[205]:


df.zindexvalue = pd.to_numeric(df.zindexvalue)


# In[159]:


df.zindexvalue.head()


# In[50]:


df.lastsolddate.min(), df.lastsolddate.max()


# The house sold period in the dateset was between January 2013 and December 2015.

# In[170]:


df.describe()


# To get a feel for the type of data we are dealing with, we plot a histogram for each numeric variable.

# In[59]:


#matplotlib inline
import matplotlib.pyplot as plt


# In[171]:


df.hist(bins = 50, figsize = (20,15))
plt.show()


# In[74]:


import seaborn as sns
sns.distplot(df.bathrooms)
plt.show()


# In[77]:


plt.figure(1)
plt.subplot(121)
sns.distplot(df.lastsoldprice)
plt.subplot(122)
df.lastsoldprice.plot.box(figsize=(16,5))
plt.show()


# In[92]:


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2)
plt.show()


# In[87]:


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7), c="lastsoldprice", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.show()


# This image tells us that the most expensive houses sold were in the north area.
# 
# The variable we are going to predict is the “last sold price”. So let’s look at how much each independent variable correlates with this dependent variable.

# In[172]:


corr_mat = df.corr()
corr_mat


# In[173]:


corr_mat['lastsoldprice'].sort_values(ascending = False)


# The last sold price tends to increase when the finished sqft and the number of bathrooms go up. We can see a small negative correlation between the year built and the last sold price. And finally, coefficients close to zero indicates that there is no linear correlation.

# We are now going to visualize the correlation between variables by using Pandas’ scatter_matrix function. We will just focus on a few promising variables, that seem the most correlated with the last sold price.

# In[105]:


from pandas.tools.plotting import scatter_matrix
attributes = ['lastsoldprice', 'finishedsqft','bathrooms','zindexvalue']
scatter_matrix(df[attributes], figsize=(16,5))
plt.show()


# The most promising variable for predicting the last sold price is the finished sqft, so let’s zoom in on their correlation scatter plot.

# In[106]:


df.plot(kind = 'scatter', x='finishedsqft', y='lastsoldprice')
plt.show()


# The correlation is indeed very strong; we can clearly see the upward trend and that the points are not too dispersed.

# Because each house has different square footage and each neighborhood has different home prices, what we really need is the price per sqft. So, we add a new variable “price_per_sqft”. We then check to see how much this new independent variable correlates with the last sold price.

# In[174]:


df['price_per_sqft'] = df['lastsoldprice']/df['finishedsqft']

corr_matrix = df.corr()
corr_matrix["lastsoldprice"].sort_values(ascending=False)


# Unfortunately, the new price_per_sqft variable shows only a very small positive correlation with the last sold price. But we still need this variable for grouping neighborhoods.

# In[114]:


len(df.neighborhood.value_counts())


# In[124]:


freq = df.groupby('neighborhood').count()['price_per_sqft']
freq.head()


# In[175]:


mean_price = df.groupby('neighborhood').mean()['price_per_sqft']
mean_price.head()


# In[132]:


cluster = pd.concat([freq, mean_price], axis = 1)
cluster.neighborhood = cluster.index
cluster.columns = ['freq', 'mean_price']
cluster.head()


# In[133]:


cluster.describe()


# These are the low price neighborhoods:

# In[141]:


cluster1 = cluster[cluster.mean_price < 756]
cluster1.index


# These are the high price and low frequency neighborhoods:

# In[139]:


cluster_temp = cluster[cluster.mean_price >= 756]
cluster2 = cluster_temp[cluster_temp.freq <123]
cluster2.index


# These are the high price and high frequency neighborhoods:

# In[140]:


cluster3 = cluster_temp[cluster_temp.freq >=123]
cluster3.index


# We add a group column based on the clusters:

# In[206]:


def get_group(x):
    if x in cluster1.index:
        return 'low_price'
    elif x in cluster2.index:
        return 'high_price_low_freq'
    else:
        return 'high_price_high_freq'
df['group'] = df.neighborhood.apply(get_group)


# After performing the above pre-processing, we do not need the following columns anymore: “address, lastsolddate, latitude, longitude, neighborhood, price_per_sqft”, so we drop them from our analysis.

# In[207]:


df.columns


# In[208]:


df.drop(df.columns[[3, 5, 6, 7]], axis=1, inplace=True)


# Gives this table:

# In[209]:


df1 = df
df = df[['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms', 'usecode', 'yearbuilt', 'zindexvalue', 'group', 'lastsoldprice']]
df.head()


# Our data looks perfect!
# 
# But before we build the model, we need to create dummy variables for these two categorical variables: “usecode” and “group”.

# In[210]:


X = df[['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms', 'usecode', 'yearbuilt', 'zindexvalue', 'group']]
Y = df['lastsoldprice']


# In[212]:


n = pd.get_dummies(df.group)
m = pd.get_dummies(df.usecode)
X = pd.concat([X, m, n], axis=1)
drops = ['group', 'usecode']
X.drop(drops, inplace=True, axis=1)
X.head()


# # Train and Build a Linear Regression Model

# In[213]:


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Done! We now have a working Linear Regression model.
# 
# Calculate R squared:

# In[214]:


y_pred = regressor.predict(X_test)
print('Liner Regression R squared: %.4f' % regressor.score(X_test, y_test))


# So, in our model, 56.19% of the variability in Y can be explained using X. This is not that exciting.
# 
# Calculate root-mean-square error (RMSE):

# In[215]:


from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(y_pred, y_test)
lin_rmse = np.sqrt(lin_mse)
print('Liner Regression RMSE: %.4f' % lin_rmse)


# Our model was able to predict the value of every house in the test set within $616071 of the real price.
# 
# Calculate mean absolute error (MAE):

# In[216]:


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(y_pred, y_test)
print('Liner Regression MAE: %.4f' % lin_mae)


# ## Random Forest

# In[217]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(X_train, y_train)


# In[218]:


print('Random Forest R squared": %.4f' % forest_reg.score(X_test, y_test))


# In[219]:


y_pred = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_pred, y_test)
forest_rmse = np.sqrt(forest_mse)
print('Random Forest RMSE: %.4f' % forest_rmse)


# Much better! Let’s try one more.

# ## Gradient boosting

# In[220]:


from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
model = ensemble.GradientBoostingRegressor()
model.fit(X_train, y_train)


# In[221]:


print('Gradient Boosting R squared": %.4f' % model.score(X_test, y_test))


# In[222]:


y_pred = model.predict(X_test)
model_mse = mean_squared_error(y_pred, y_test)
model_rmse = np.sqrt(model_mse)
print('Gradient Boosting RMSE: %.4f' % model_rmse)


# These are the best results we got so far, so, I would consider this is our final model.

# ##### Feature Importance

# We have used 19 features (variables) in our model. Let’s find out which features are important and vice versa.

# In[224]:


feature_labels = np.array(['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms', 'yearbuilt', 'zindexvalue', 
                           'high_price_high_freq', 'high_price_low_freq', 'low_price', 'Apartment', 'Condominium', 'Cooperative', 
                          'Duplex', 'Miscellaneous', 'Mobile', 'MultiFamily2To4', 'MultiFamily5Plus', 'SingleFamily', 
                           'Townhouse'])
importance = model.feature_importances_
importance


# In[225]:


feature_indexes_by_importance = importance.argsort()
feature_indexes_by_importance


# In[230]:


for index in feature_indexes_by_importance:
    print('{} - {:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))


# The most important features are finished sqft, zindex value, number of bathrooms, total rooms, year built and so on. And the least important feature is Apartment, which means that regardless of whether this unit is an apartment or not, does not matter to the sold price. Overall, most of these 19 features are used.
