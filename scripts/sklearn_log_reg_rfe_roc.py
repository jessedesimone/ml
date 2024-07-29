# Import packages
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Load data from CSV
file_path = 'Book21.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Define predictors and target
X = df.iloc[:, 1:]  # Assuming all columns except the first are predictors
y = df.iloc[:, 0]  # Assuming the first column is the target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform recursive feature elimination
# Initialize logistic regression model
logreg = LogisticRegression(max_iter=1000, solver='saga')

# Initialize RFE with logistic regression model and desired number of features
rfe = RFE(logreg, n_features_to_select=10)

# Fit RFE
rfe = rfe.fit(X_train_scaled, y_train)

# Get the selected features
selected_features = X.columns[rfe.support_]
print('Selected features:', selected_features)

# Fit the final model with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Scale the selected features
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Train the logistic regression model using statsmodels
X_train_const = sm.add_constant(X_train_selected_scaled)
final_model = sm.Logit(y_train, X_train_const).fit()
print(final_model.summary())

# Predict probabilities on the test set
X_test_const = sm.add_constant(X_test_selected_scaled)
y_pred_proba = final_model.predict(X_test_const)

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Plot feature importance based on logistic regression coefficients
coefficients = final_model.params[1:]  # Exclude the intercept
coefficients = coefficients.sort_values()

plt.figure(figsize=(10, 6))
coefficients.plot(kind='barh')
plt.title('Feature Importance in Logistic Regression Model')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()
