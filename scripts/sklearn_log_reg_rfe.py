#!/usr/bin/env python3

'''
sklearn logistic regression with recursive feature elimination
'''

# Import packages
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data from Excel
file_path = 'cleaned_df.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Define predictors and target
X = df.iloc[:, 1:]  # Assuming all columns except the first are predictors
y = df.iloc[:, 0]  # Assuming the first column is the target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform recursive feature elimination
# Initialize logistic regression model
logreg = LogisticRegression(max_iter=500, solver='saga')

# Initialize RFE with logistic regression model and desired number of features
rfe = RFE(logreg, n_features_to_select=10)

# Fit RFE
rfe = rfe.fit(X, y)

# Get the selected features
selected_features = X.columns[rfe.support_]
print('Selected features:', selected_features)

# Fit the final model with selected features
X_selected = X[selected_features]
X_selected_scaled = scaler.fit_transform(X_selected)
final_model = sm.Logit(y, sm.add_constant(X_selected_scaled)).fit()
print(final_model.summary())

# Plot feature importance based on logistic regression coefficients
coefficients = final_model.params[1:]  # Exclude the intercept
coefficients = coefficients.sort_values()

plt.figure(figsize=(10, 6))
coefficients.plot(kind='barh')
plt.title('Feature Importance in Logistic Regression Model')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()