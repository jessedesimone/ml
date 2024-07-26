#!/usr/bin/env python3

'''
Stepwise logistic regression model
Stepwise regression involves iteratively adding or removing predictors based on specific criteria like p-values or AIC (Akaike Information Criterion)

OPTION 1: standard model
OPTION 2: define maximum features
OPTION 3: define minimum features
'''

## OPTION 1
# Import packages
import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Set wd
os.chdir('/Users/jessedesimone/Desktop')

# Load data from Excel
file_path = 'cleaned_df.csv'  # Replace with your file path
df = pd.read_csv(file_path)

TARGET='CLIN_CONVERT'             #define the target variable

# Define the plotting function for feature importance
def plot_feat_importance():
    coefficients = final_model.params[1:]  # Exclude the intercept
    coefficients = coefficients.sort_values()
    plt.figure(figsize=(10, 6))
    coefficients.plot(kind='barh')
    plt.title('Feature Importance in Logistic Regression Model', weight='bold')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

# Define predictors and target
X = df.iloc[:, 1:]  # Assuming all columns except the first are predictors
y = df.iloc[:, 0]  # Assuming the first column is the target variable

# # Define stepwise function
# def stepwise_selection(X, y, 
#                        initial_list=[], 
#                        threshold_in=0.01, 
#                        threshold_out=0.05,
#                        verbose=True):
#     """ Perform a forward-backward feature selection 
#     based on p-value from statsmodels.api.OLS
#     Arguments:
#         X - pandas.DataFrame with candidate features
#         y - list-like with the target
#         initial_list - list of features to start with (column names of X)
#         threshold_in - include a feature if its p-value < threshold_in
#         threshold_out - exclude a feature if its p-value > threshold_out
#         verbose - whether to print the sequence of inclusions and exclusions
#     Returns: list of selected features 
#     Always set threshold_in < threshold_out to avoid infinite loop.
#     """
#     included = list(initial_list)
#     while True:
#         changed = False
#         # forward step
#         excluded = list(set(X.columns) - set(included))
#         new_pval = pd.Series(index=excluded)
#         for new_column in excluded:
#             model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(disp=0)
#             new_pval[new_column] = model.pvalues[new_column]
#         best_pval = new_pval.min()
#         if best_pval < threshold_in:
#             best_feature = new_pval.idxmin()
#             included.append(best_feature)
#             changed = True
#             if verbose:
#                 print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        
#         # backward step
#         model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
#         # use all coefs except the intercept
#         pvalues = model.pvalues.iloc[1:]
#         worst_pval = pvalues.max()
#         if worst_pval > threshold_out:
#             changed = True
#             worst_feature = pvalues.idxmax()
#             included.remove(worst_feature)
#             if verbose:
#                 print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
#         if not changed:
#             break
    
#     return included

# # Apply the stepwise function
# # Define predictors and target
# X = df.drop(columns=[TARGET])
# y = df[TARGET]

# # Perform stepwise logistic regression
# result = stepwise_selection(X, y)
# print('Selected features:', result)

# # Fit the final model with selected features
# final_model = sm.Logit(y, sm.add_constant(X[result])).fit()
# print(final_model.summary())

# # Plot feature importance
# plot_feat_importance()

# '''
# OPTION 2
# New model with predefined set of max final features e.g., up to 10 most important features
# '''

# def stepwise_selection(X, y, 
#                        initial_list=[], 
#                        threshold_in=0.01, 
#                        threshold_out=0.05, 
#                        max_features=10,
#                        verbose=True):
#     """ Perform a forward-backward feature selection 
#     based on p-value from statsmodels.api.OLS
#     Arguments:
#         X - pandas.DataFrame with candidate features
#         y - list-like with the target
#         initial_list - list of features to start with (column names of X)
#         threshold_in - include a feature if its p-value < threshold_in
#         threshold_out - exclude a feature if its p-value > threshold_out
#         max_features - maximum number of features to include
#         verbose - whether to print the sequence of inclusions and exclusions
#     Returns: list of selected features 
#     Always set threshold_in < threshold_out to avoid infinite loop.
#     """
#     included = list(initial_list)
#     while True:
#         changed = False
#         # forward step
#         excluded = list(set(X.columns) - set(included))
#         new_pval = pd.Series(index=excluded)
#         for new_column in excluded:
#             model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(disp=0)
#             new_pval[new_column] = model.pvalues[new_column]
#         best_pval = new_pval.min()
#         if best_pval < threshold_in and len(included) < max_features:
#             best_feature = new_pval.idxmin()
#             included.append(best_feature)
#             changed = True
#             if verbose:
#                 print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        
#         model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
#         # use all coefs except the intercept
#         pvalues = model.pvalues.iloc[1:]
#         worst_pval = pvalues.max()
#         if worst_pval > threshold_out:
#             changed = True
#             worst_feature = pvalues.idxmax()
#             included.remove(worst_feature)
#             if verbose:
#                 print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
#         if not changed:
#             break
#     return included[:max_features]  # Ensure we only return up to max_features

# # Apply the stepwise function
# # Define predictors and target
# X = df.drop(columns=[TARGET])
# y = df[TARGET]

# # Perform stepwise logistic regression
# result = stepwise_selection(X, y, max_features=10)
# print('Selected features:', result)

# # Fit the final model with selected features
# final_model = sm.Logit(y, sm.add_constant(X[result])).fit()
# print(final_model.summary())

# # Plot feature importance
# plot_feat_importance()

'''
OPTION 3
New model with predefined set of MIN final features e.g., minimum 10 most important features
'''

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out=0.05, 
                       min_features=10,
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        min_features - minimum number of features to include
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite loop.
    """
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in or len(included) < min_features:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        # backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
        # use all coefs except the intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out and len(included) > min_features:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

# Apply the stepwise function
# Define predictors and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Perform stepwise logistic regression
result = stepwise_selection(X, y, min_features=10)
print('Selected features:', result)

# Fit the final model with selected features
final_model = sm.Logit(y, sm.add_constant(X[result])).fit()
print(final_model.summary())

# Plot feature importance
plot_feat_importance()

