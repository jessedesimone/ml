'''
source: https://www.w3schools.com/python/python_ml_multiple_regression.asp
'''

import pandas as pd
from sklearn import linear_model

inpath = '/<path/to/file>'
df = pd.read_csv(inpath + 'FUELCONSUMPTION.csv')

X = df[['CYLINDERS']]
y = df['CO2EMISSIONS']

regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict CO2 emissions for car with 5 cylinders
predictedCO2 = regr.predict([[5]])

print(predictedCO2)