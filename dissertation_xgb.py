#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve


# # Processing

# In[2]:


# Load the previously uploaded CSV files again
output_df = pd.read_csv('F:/dissertation/output.csv')
list_df = pd.read_csv('F:/dissertation/list.csv')

# Drop columns with too many missing values or that are not needed for the model
columns_to_drop = ['Note3','page','Publication Year']
output_df_cleaned = output_df.drop(columns=columns_to_drop)

# Fill missing values with a placeholder or appropriate values
output_df_cleaned['Note'].fillna('Unknown', inplace=True)
output_df_cleaned['Note1'].fillna('Unknown', inplace=True)
output_df_cleaned['Note2'].fillna('Unknown', inplace=True)

# Create a binary target variable indicating if the Identifier is in the list
output_df_cleaned['Target'] = output_df_cleaned['Identifier'].isin(list_df.iloc[:, 0]).astype(int)

# Select the relevant features for conglomerate classification
selected_features = [
    'Class','Note','Space Gp. Number', 'Number of chiral center', 'Number of Carbon Chiral Atom',
    'Number of Chiral Center having H', 'R', 'S', 'M', 'a', 'b', 'c', 'Alpha', 'Beta',
    'Gamma', 'Cell Volume', 'Calc. Density', 'Z Value', 'R-factor', 'Has Disorder?']

# Filter the dataset with selected features and the target variable
df_model = output_df_cleaned[selected_features + ['Target']].copy()

# Convert categorical columns to numerical using label encoding
label_encoder = LabelEncoder()

# Encode boolean columns and other categorical variables
df_model['Has Disorder?'] = label_encoder.fit_transform(df_model['Has Disorder?'])
df_model['Space Gp. Number'] = label_encoder.fit_transform(df_model['Space Gp. Number'])
df_model['Class'] = label_encoder.fit_transform(df_model['Class'])
df_model['Note'] = label_encoder.fit_transform(df_model['Note'])

# Prepare the features (X) and the target (y)
X = df_model.drop('Target', axis=1)
y = df_model['Target']

# Display the first few rows to check if the data is loaded correctly
df_model.head()


# # XGBoost model with resampled dataset

# In[3]:



from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model with resampled data
xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_resampled = xgb_model.predict(X_test_resampled)

# Evaluate the model's performance
accuracy_resampled = accuracy_score(y_test_resampled, y_pred_resampled)
report_resampled = classification_report(y_test_resampled, y_pred_resampled)

# Plot the confusion matrix for the resampled model
conf_matrix_resampled = confusion_matrix(y_test_resampled, y_pred_resampled)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_resampled, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix with SMOTE and Class Weight Adjustment')
plt.show()

print(accuracy_resampled, report_resampled)

# Check feature importances from the trained Random Forest model
feature_importances = xgb_model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in XGBoost Model')
plt.gca().invert_yaxis()
plt.show()

importance_df


# In[4]:


#performance on real dataset
y_pred_proba = xgb_model.predict_proba(X)[:, 1]
# Calculate precision and recall for different thresholds
precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)

# Find the threshold that balances precision and recall
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold_index = f1_scores.argmax()
best_threshold = thresholds[best_threshold_index]

# Adjust the decision threshold to the optimal value found
y_pred_best_threshold = (y_pred_proba >= best_threshold).astype(int)

# Evaluate model performance with the optimal threshold
accuracy_best_threshold = accuracy_score(y, y_pred_best_threshold)
conf_matrix_best_threshold = confusion_matrix(y, y_pred_best_threshold)
class_report_best_threshold = classification_report(y, y_pred_best_threshold)

print(best_threshold, accuracy_best_threshold, conf_matrix_best_threshold, class_report_best_threshold)


# In[5]:


#Precision-recall curve
# Calculate F1 scores for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall)

# Find the index of the maximum F1 score
best_f1_index = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_index]

# Calculate the distance to (1,1) for each point on the PR curve
distances_to_one_one = np.sqrt((1 - precision)**2 + (1 - recall)**2)

# Find the index of the smallest distance to (1,1)
best_distance_index = np.argmin(distances_to_one_one)
best_distance_threshold = thresholds[best_distance_index]

# Plot the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='XGBoost')
plt.scatter(recall[best_f1_index], precision[best_f1_index], color='red', label=f'Best F1 Threshold ({best_f1_threshold:.2f})', s=100)
plt.scatter(recall[best_distance_index], precision[best_distance_index], color='blue', label=f'Closest to (1,1) Threshold ({best_distance_threshold:.2f})', s=100)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()


# In[6]:


# Adjust the decision threshold to increase specificity (reduce false positives)
threshold = best_f1_threshold   # Increase the threshold to make the model more conservative
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Evaluate model performance with adjusted threshold
accuracy_adjusted = accuracy_score(y, y_pred_adjusted)
conf_matrix_adjusted = confusion_matrix(y, y_pred_adjusted)
class_report_adjusted = classification_report(y, y_pred_adjusted)

# Display the results
print(accuracy_adjusted, conf_matrix_adjusted, class_report_adjusted)


# In[7]:


#reproduce the list and compare with actual list
# Extract identifiers predicted as conglomerates by the model
conglomerate_identifiers = X.index[y_pred_proba >= threshold]

# Map indices back to Identifiers in the original dataframe
predicted_conglomerates = output_df.loc[conglomerate_identifiers, 'Identifier'].unique()

# Display the predicted conglomerate identifiers
print(predicted_conglomerates)


# Compare the model's predicted conglomerate identifiers with the actual conglomerates in list_df
actual_conglomerates = set(list_df['Identifier'])
predicted_conglomerates_set = set(predicted_conglomerates)

# Find the intersection and differences between predicted and actual conglomerates
correct_predictions = actual_conglomerates.intersection(predicted_conglomerates_set)
false_positives = predicted_conglomerates_set - actual_conglomerates
missed_conglomerates = actual_conglomerates - predicted_conglomerates_set

# Display the results
{
    "Correct Predictions (True Positives)": len(correct_predictions),
    "False Positives (Incorrect Predictions)": len(false_positives),
    "Missed Conglomerates (False Negatives)": len(missed_conglomerates)
}


# # XGboost model using new input

# In[8]:


# Select the relevant features for conglomerate classification
new_features = [
    'Class','Note','Number of chiral center', 
    'Number of Chiral Center having H', 'R', 'S','Has Disorder?']

# Filter the dataset with selected features and the target variable
df_model2 = output_df_cleaned[new_features + ['Target']].copy()

# Display the first few rows to check if the data is loaded correctly
df_model2.head()


# In[9]:


# Convert categorical columns to numerical using label encoding
label_encoder = LabelEncoder()

# Encode boolean columns and other categorical variables
df_model2['Has Disorder?'] = label_encoder.fit_transform(df_model2['Has Disorder?'])
df_model2['Class'] = label_encoder.fit_transform(df_model2['Class'])
df_model2['Note'] = label_encoder.fit_transform(df_model2['Note'])

# Prepare the features (X) and the target (y)
X = df_model2.drop('Target', axis=1)
y = df_model2['Target']


# In[10]:


# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model with resampled data
xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_resampled = xgb_model.predict(X_test_resampled)

# Evaluate the model's performance
accuracy_resampled = accuracy_score(y_test_resampled, y_pred_resampled)
report_resampled = classification_report(y_test_resampled, y_pred_resampled)

# Plot the confusion matrix for the resampled model
conf_matrix_resampled = confusion_matrix(y_test_resampled, y_pred_resampled)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_resampled, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix with SMOTE and Class Weight Adjustment')
plt.show()

print(accuracy_resampled, report_resampled)

# Check feature importances from the trained Random Forest model
feature_importances = xgb_model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in new XGBoost Model')
plt.gca().invert_yaxis()
plt.show()

importance_df


# In[11]:


y_pred_proba = xgb_model.predict_proba(X)[:, 1]
# Calculate precision and recall for different thresholds
precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)

# Find the threshold that balances precision and recall
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold_index = f1_scores.argmax()
best_threshold = thresholds[best_threshold_index]

# Adjust the decision threshold to the optimal value found
y_pred_best_threshold = (y_pred_proba >= best_threshold).astype(int)

# Evaluate model performance with the optimal threshold
accuracy_best_threshold = accuracy_score(y, y_pred_best_threshold)
conf_matrix_best_threshold = confusion_matrix(y, y_pred_best_threshold)
class_report_best_threshold = classification_report(y, y_pred_best_threshold)

best_threshold, accuracy_best_threshold, conf_matrix_best_threshold, class_report_best_threshold


# In[12]:


# Calculate F1 scores for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall)

# Find the index of the maximum F1 score
best_f1_index = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_index]

# Calculate the distance to (1,1) for each point on the PR curve
distances_to_one_one = np.sqrt((1 - precision)**2 + (1 - recall)**2)

# Find the index of the smallest distance to (1,1)
best_distance_index = np.argmin(distances_to_one_one)
best_distance_threshold = thresholds[best_distance_index]

# Plot the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='XGBoost')
plt.scatter(recall[best_f1_index], precision[best_f1_index], color='red', label=f'Best F1 Threshold ({best_f1_threshold:.2f})', s=100)
plt.scatter(recall[best_distance_index], precision[best_distance_index], color='blue', label=f'Closest to (1,1) Threshold ({best_distance_threshold:.2f})', s=100)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()


# In[13]:


# Adjust the default threshold to increase specificity (reduce false positives)
threshold = 0.5   # Increase the threshold to make the model more conservative
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Evaluate model performance with adjusted threshold
accuracy_adjusted = accuracy_score(y, y_pred_adjusted)
conf_matrix_adjusted = confusion_matrix(y, y_pred_adjusted)
class_report_adjusted = classification_report(y, y_pred_adjusted)

# Display the results
print(accuracy_adjusted, conf_matrix_adjusted, class_report_adjusted)


# In[14]:


# Adjust the decision threshold to increase specificity (reduce false positives)
threshold = 0.8  # Increase the threshold to make the model more conservative
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Evaluate model performance with adjusted threshold
accuracy_adjusted = accuracy_score(y, y_pred_adjusted)
conf_matrix_adjusted = confusion_matrix(y, y_pred_adjusted)
class_report_adjusted = classification_report(y, y_pred_adjusted)

# Display the results
print(accuracy_adjusted, conf_matrix_adjusted, class_report_adjusted)


# In[15]:


#Reproduce the list and comparison
# Extract identifiers predicted as conglomerates by the model
conglomerate_identifiers = X.index[y_pred_proba >= threshold]

# Map indices back to Identifiers in the original dataframe
predicted_conglomerates = output_df.loc[conglomerate_identifiers, 'Identifier'].unique()

# Display the predicted conglomerate identifiers
print(predicted_conglomerates)


# Compare the model's predicted conglomerate identifiers with the actual conglomerates in list_df
actual_conglomerates = set(list_df['Identifier'])
predicted_conglomerates_set = set(predicted_conglomerates)

# Find the intersection and differences between predicted and actual conglomerates
correct_predictions = actual_conglomerates.intersection(predicted_conglomerates_set)
false_positives = predicted_conglomerates_set - actual_conglomerates
missed_conglomerates = actual_conglomerates - predicted_conglomerates_set

# Display the results
{
    "Correct Predictions (True Positives)": len(correct_predictions),
    "False Positives (Incorrect Predictions)": len(false_positives),
    "Missed Conglomerates (False Negatives)": len(missed_conglomerates)
}


# In[ ]:




