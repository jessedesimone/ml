#!/usr/local/bin/python3.9
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

inpath = '<file path>'
df = pd.read_csv(inpath + '<infile>')

df.shape
df.info()
df.head()
df.describe()

x=df['sqft_living']
y=df['price']

x = np.array(y).reshape(-1, 1)
y = np.array(y)

#Splitting the data into Train and Test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
print('linear equation = ', regressor.coef_, 'x', regressor.intercept_)

#Predicting the prices
pred = regressor.predict(xtest)
#Create dataframe of predicted vs actual values
df2 = pd.DataFrame({'Actual': ytest, 'Predicted': pred})
df2

#Visualizing the training Test Results
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#Visualizing the training Test Results
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Predict new value
new_X = 4000
X = np.array(new_X).reshape(-1,1)
regressor.predict(X)


'''
Example 2
Source: https://www.kaggle.com/mrizwanse/simple-linear-regression-with-sklearn-salarydata
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import data
inpath = '<path/to/file>'
df = pd.read_csv(inpath + 'FuelConsumption.csv')

# Examine data
df.info()
df.describe()

# Visualize the data using heatmap
sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.tight_layout()
plt.show()

'''
Engine size is highly correlated with CO2 emissions. Let's
select this variable as for X, CO2 emissions for Y
'''
X = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS',
             'CYLINDERS','TRANSMISSION', 'FUELTYPE',
             'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY',
             'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG',
             'CO2EMISSIONS'], axis=1).values
y = df['CO2EMISSIONS']
X.shape, y.shape

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Fitting simple linear regression to the Training Set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print('linear equation = ', regressor.coef_, 'x', regressor.intercept_)

# Fitting the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train).fit()
lr.summary()

#Visualizing the training Test Results
plt.scatter(X_train, y_train, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

# Predicting the values of the test set
pred = regressor.predict(X_test)

# Create dataframe of predicted vs actual values
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
df2

#Visualizing the training Test Results
plt.scatter(X_test, y_test, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title ("Visuals for Testing Dataset")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print('Model coef:', regressor.coef_)
print('Model Intercept:', regressor.intercept_)
print('Equation:', regressor.coef_, '* new value of X', regressor.intercept_)
print('R^2 score:', r2_score(y_test, pred))

# Predictions on new values using the trained model
'''
Predict the CO2 emissions associated with an engine size of 6.5
'''
new_value = 6.5
print('Predicted CO2 Emissions:',regressor.predict([[new_value]]))
print('Predicted CO2 Emissions:', (regressor.coef_)*(new_value)+(regressor.intercept_))