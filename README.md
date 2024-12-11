# MLproject

import zipfile
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score,precision_recall_curve, auc, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform


# Extracting Dataset Credit Card Fraud Detection
zip_path = r'C:\Users\Hp\Desktop\SEM 4TH\AI PROGRAMMING\archive.zip'
extract_to = r'C:\Users\Hp\Desktop\SEM 4TH\AI PROGRAMMING\archive'

if os.path.exists(zip_path):
    print("ZIP file found. Extracting...")
    # Step 2: Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)  
        print("Extraction complete!")
else:
    print("ZIP file not found. Please check the path.")

dataset_path = os.path.join(extract_to, 'creditcard.csv')  
if os.path.exists(dataset_path):
    print("Dataset found. Loading...")
    data = pd.read_csv(dataset_path)
    print("Data loaded successfully!")
    print(data.head())  
else:
    print("Dataset not found. Ensure the ZIP file was extracted correctly.")


# Data Cleaning
print(data.info())
print(data.isnull().sum())

# Dropping rows with missing values (if any)
data = data.dropna()

# Checking for duplicate rows
print("\nDuplicate Rows:", data.duplicated().sum())
# removing duplicate rows
data = data.drop_duplicates()
print(f"New shape after removing duplicates: {data.shape}")



# # 4. Handle outliers (e.g., using IQR for 'Amount')
Q1 = data['Amount'].quantile(0.25)
Q3 = data['Amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping extreme values
data['Amount'] = data['Amount'].clip(lower=lower_bound, upper=upper_bound)

# 5. Feature scaling (standardizing 'Amount' and 'Time')
scaler = StandardScaler()
data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

#  Visualizing 'Amount' for fraud and non-fraud transactions
sns.boxplot(x='Class', y='Amount', data=data)
plt.title("Amount Distribution by Class")
plt.show()

# splitting features and targets
x=data.drop(columns=['Class'])
y=data['Class']

# train_test split
x_train, x_test, y_train, y_test=train_test_split(x, y,test_size=0.2, random_state=42, stratify=y)

fraud_train = x_train[y_train == 1]
non_fraud_train = x_train[y_train == 0]

fraud_oversampled = fraud_train.sample(len(non_fraud_train), replace=True, random_state=42)
x_train_balanced = pd.concat([non_fraud_train, fraud_oversampled])
y_train_balanced = pd.concat([y_train[y_train == 0], y_train[y_train == 1].sample(len(non_fraud_train), replace=True, random_state=42)])

x_train_balanced, y_train_balanced = sklearn.utils.shuffle(x_train_balanced, y_train_balanced, random_state=42)
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

x_train = pd.DataFrame(x_train, columns=x.columns)
x_test = pd.DataFrame(x_test, columns=x.columns)

# plotting AUC-ROC and AUC-PR curves
def plot_roc_and_pr_curve(model, x_test, y_test, model_name):
    # ROC Curve
    y_prob = model.predict_proba(x_test)[:, 1]  
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    # Plot ROC Curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")

    # Plot PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

# NAIVE BAYES ALGORITHM
print("\nTraining Naive Bayes Model\n")
nb_model=GaussianNB()
nb_model.fit(x_train_balanced, y_train_balanced)
nb_pred=nb_model.predict(x_test)
print("Result by Naive Bayes: \n")
print(f"Accuracy: {accuracy_score(y_test, nb_pred):.2f}")
print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred))
plot_roc_and_pr_curve(nb_model, x_test, y_test, "Naive Bayes")

# random search for Naive Bayes
param_grid_nb = {'var_smoothing': np.logspace(-9, 0, 50)}
nb_random_search = RandomizedSearchCV(estimator=GaussianNB(), param_distributions=param_grid_nb, scoring='accuracy', n_iter=20, cv=5, random_state=42)
nb_random_search.fit(x_train, y_train)

print("Best Parameters for Naive Bayes:", nb_random_search.best_params_)
optimized_nb = nb_random_search.best_estimator_
y_pred_nb_opt = optimized_nb.predict(x_test)
print("Optimized Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb_opt))

# # LOGISTIC REGRESSION ALGORITHM
print("\nTraining Logistic Regression Model\n")
lr_model=LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(x_train_balanced, y_train_balanced)
lr_pred=lr_model.predict(x_test)
print("Logistic Regression Results\n")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.2f}")
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
plot_roc_and_pr_curve(nb_model, x_test, y_test, "Logistic Regression")

# cross validation for logistic Regression
lr_cv_scores=cross_val_score(lr_model, x, y, cv=5, scoring='accuracy')
accuracy=lr_cv_scores.mean()
std_dev=lr_cv_scores.std()
print(f"Cross Validation Scores for Logistic Regression: {accuracy:.2f} ± {std_dev:.2f} ")

# Logistic Regression Hyperparameter Tuning
param_grid_lr = {
    'C': np.logspace(-3, 3, 50),
    'solver': ['liblinear', 'lbfgs'],  # Use only compatible solvers
    'penalty': ['l2'],
    'max_iter': [100, 500, 1000]
}
lr_random_search = RandomizedSearchCV(estimator=LogisticRegression(random_state=42), param_distributions=param_grid_lr, scoring='accuracy', n_iter=20, cv=5, random_state=42, error_score='raise')
lr_random_search.fit(x_train, y_train)

print("Best Parameters for Logistic Regression:", lr_random_search.best_params_)
optimized_lr = lr_random_search.best_estimator_
y_pred_lr_opt = optimized_lr.predict(x_test)
print("Optimized Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr_opt))



# RANDOM FOREST ALGORITHM
print("\nTraining Random Forest Model\n")
rf_model=RandomForestClassifier(random_state=42, n_estimators=10, max_depth=10)
rf_model.fit(x_train_balanced, y_train_balanced)
rf_pred=rf_model.predict(x_test)
print("RANDOM FOREST RESULTS: \n")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print(confusion_matrix(y_test,rf_pred))
print(classification_report(y_test, rf_pred))

# feature importance for Random Forest 
feature_importance=rf_model.feature_importances_
features=data.drop(columns=['Class']).columns
plt.figure(figsize=(10,6))
plt.barh(features, feature_importance, color='green')
plt.xlabel("Feature Importance")
plt.title("Random Forest Importance")
plt.show()


# OPTIMIZING HYPERPARAMETER WITH GRID SEARCH FOR RANDOM FOREST
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}
rf_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid_rf,
    scoring='accuracy',
    cv=5
)
rf_grid_search.fit(x_train, y_train)
optimized_rf_grid = rf_grid_search.best_estimator_

# evaluation on test sets
y_pred_grid_opt = optimized_rf_grid.predict(x_test)
print("\nGrid Search Optimized Random Forest Results on Test Data:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_grid_opt):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_grid_opt))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_grid_opt))


# Randomized Search for Random Forest
rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_grid_rf,
    scoring='accuracy',
    n_iter=20,
    cv=5,
    random_state=42
)
rf_random_search.fit(x_train, y_train)
optimized_rf_random = rf_random_search.best_estimator_

# Evaluation on the test set for Random Search
y_pred_random_opt = optimized_rf_random.predict(x_test)
print("\nRandom Search Optimized Random Forest Results on Test Data:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_random_opt):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_random_opt))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_random_opt))
    

# Testing optimized models
optimized_models = [
(optimized_nb, "Naive Bayes (Optimized)"),
(optimized_lr, "Logistic Regression (Optimized)"),
(optimized_rf_random, "Random Forest (Optimized)")
]

# Evaluation of each optimized model
for model, name in optimized_models:
    print(f"\n{name} Results on Test Data:")
    y_pred = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# Precision: How many predicted fraud cases are correct?
# Recall: How many actual fraud cases were detected?
# F1-Score: Balance between precision and recall.
print(f"Precision: {precision_score(y_test, rf_pred):.2f}")
print(f"Recall: {recall_score(y_test, rf_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, rf_pred):.2f}")


    

    
