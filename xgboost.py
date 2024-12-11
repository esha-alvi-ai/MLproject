#esha alvi sp23-bai-015



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load the 
csv_file_path = "/content/creditcard.csv"  # Adjust path if necessary
try:
    data = pd.read_csv(csv_file_path)
    print("\nDataset loaded successfully!")
except FileNotFoundError:
    print("CSV file not found. Ensure the file path is correct.")
    exit()

# Select important features and target
selected_features = ['V14', 'V12', 'V10', 'V16', 'Amount']  # Choose relevant features
data = data[selected_features + ['Class']]

# Preprocess data
data = data.dropna().drop_duplicates()
X = data[selected_features]
y = data['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for balancing the dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train XGBoost model
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_smote, y_train_smote)

# Evaluate the model
y_pred = xgb_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Compute Precision-Recall Curve
y_prob = xgb_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
plt.fill_between(recall, precision, color='blue', alpha=0.2)
plt.title("Precision-Recall Curve - XGBoost")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.show()

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion_matrix(y_test, y_pred)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Initialize the XGBoost model
xgb_base = XGBClassifier(random_state=42, eval_metric='logloss')


grid_search = GridSearchCV(estimator=xgb_base, param_grid=param_grid, 
                           scoring='f1', cv=3, verbose=2, n_jobs=-1)

# Perform Grid Search
print("\nStarting Grid Search for hyperparameter tuning...")
grid_search.fit(X_train_smote, y_train_smote)


best_xgb_model = grid_search.best_estimator_
print(f"\nBest Parameters from Grid Search:\n{grid_search.best_params_}")

# Evaluate the best model
y_pred = best_xgb_model.predict(X_test)
print("\nClassification Report with Tuned Model:")
print(classification_report(y_test, y_pred))

y_prob = best_xgb_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue')
plt.fill_between(recall, precision, color='blue', alpha=0.2)
plt.title("Precision-Recall Curve - Tuned XGBoost")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.show()

