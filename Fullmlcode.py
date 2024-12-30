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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score,precision_recall_curve, auc, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform


# Extracting Dataset Credit Card Fraud Detection
zip_path = r"C:\Users\ABC\Downloads\archive (1).zip"
extract_to = r"C:\Users\ABC\Downloads\archive (1)"

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
plt.figure(figsize=(15, 10))
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
print("\nRandomized Search for Naive Bayes Model\n")

nb_param_dist = {
    'var_smoothing': uniform(1e-9, 1e-3)  # Adjust the range based on experimentation
}

nb_random_search = RandomizedSearchCV(estimator=GaussianNB(),
                                      param_distributions=nb_param_dist,
                                      n_iter=10,
                                      cv=5,
                                      random_state=42,
                                      scoring='accuracy')

nb_random_search.fit(x_train_balanced, y_train_balanced)

print("Best Parameters for Naive Bayes:", nb_random_search.best_params_)
best_nb_model = nb_random_search.best_estimator_
nb_pred_optimized = best_nb_model.predict(x_test)

print(f"Optimized Naive Bayes Accuracy: {accuracy_score(y_test, nb_pred_optimized):.2f}")



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


accuracy_lr = accuracy_score(y_test, lr_pred)
precision_lr = precision_score(y_test, lr_pred)
recall_lr = recall_score(y_test, lr_pred)
f1_lr = f1_score(y_test, lr_pred)
conf_matrix_lr = confusion_matrix(y_test, lr_pred)

print(f"Cross Validation Scores for Logistic Regression: {accuracy:.2f} ± {std_dev:.2f} ")
plt.figure(figsize=(8, 6))
metrics_lr = [precision_lr, recall_lr, f1_lr]
metrics_names = ['Precision', 'Recall', 'F1-Score']
plt.bar(metrics_names, metrics_lr, color=['blue', 'orange', 'green'])
plt.title('Logistic Regression Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()

# Visualize Confusion Matrix for Logistic Regression
conf_matrix_lr = confusion_matrix(y_test, lr_pred)
plt.figure(figsize=(6, 4))
plt.show(conf_matrix_lr, cmap='Blues', fignum=1)
plt.title('Logistic Regression Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Class 0 (Non-Fraud)', 'Class 1 (Fraud)'])
plt.yticks([0, 1], ['Class 0 (Non-Fraud)', 'Class 1 (Fraud)'])
plt.show()
# Random Search for Logistic Regression
# Logistic Regression Hyperparameter Tuning
print("\nRandomized Search for Logistic Regression Model\n")

lr_param_dist = {
    'C': uniform(0.01, 10),  # Regularization strength
    'penalty': ['l2', 'elasticnet'],  # Regularization types
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],  # Optimization algorithms
    'max_iter': [1000, 2000]  # Iterations for convergence
}

lr_random_search = RandomizedSearchCV(estimator=LogisticRegression(random_state=42),
                                      param_distributions=lr_param_dist,
                                      n_iter=10,
                                      cv=5,
                                      random_state=42,
                                      scoring='accuracy')

lr_random_search.fit(x_train_balanced, y_train_balanced)

print("Best Parameters for Logistic Regression:", lr_random_search.best_params_)
best_lr_model = lr_random_search.best_estimator_
lr_pred_optimized = best_lr_model.predict(x_test)

print(f"Optimized Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred_optimized):.2f}")



# RANDOM FOREST ALGORITHM
print("\nTraining Random Forest Model\n")
rf_model=RandomForestClassifier(random_state=42, n_estimators=10, max_depth=10)
rf_model.fit(x_train_balanced, y_train_balanced)
rf_pred=rf_model.predict(x_test)
print("RANDOM FOREST RESULTS for imbalanced data: \n")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print(confusion_matrix(y_test,rf_pred))
print(classification_report(y_test, rf_pred))

cm_rf = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='Blues', values_format='*')

# Show the plot
plt.show()

# feature importance for Random Forest 
feature_importance=rf_model.feature_importances_
features=data.drop(columns=['Class']).columns
plt.figure(figsize=(10,6))
plt.barh(features, feature_importance, color='green')
plt.xlabel("Feature Importance")
plt.title("Random Forest Importance")
plt.show()


# OPTIMIZING HYPERPARAMETER WITH GRID SEARCH FOR RANDOM FOREST
rf_param = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=rf_param,
                              cv=3,
                              scoring='accuracy')
grid_search_rf.fit(x_train, y_train)
print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
best_rf = grid_search_rf.best_estimator_
rf_pred_optimized = best_rf.predict(x_test)
print(f"Optimized Random Forest Accuracy: {accuracy_score(y_test, rf_pred_optimized):.2f}")

# Random Search forf Random Forest
print("\nRandomized Search for Random Forest Model\n")

rf_param_dist = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                      param_distributions=rf_param_dist,
                                      n_iter=20,
                                      cv=5,
                                      random_state=42,
                                      scoring='accuracy')

rf_random_search.fit(x_train_balanced, y_train_balanced)

print("Best Parameters for Random Forest:", rf_random_search.best_params_)
best_rf_model = rf_random_search.best_estimator_
rf_pred_optimized = best_rf_model.predict(x_test)

print(f"Optimized Random Forest Accuracy: {accuracy_score(y_test, rf_pred_optimized):.2f}")


# Precision: How many predicted fraud cases are correct?
# Recall: How many actual fraud cases were detected?
# F1-Score: Balance between precision and recall.
print(f"Precision: {precision_score(y_test, rf_pred):.2f}")
print(f"Recall: {recall_score(y_test, rf_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, rf_pred):.2f}")

# testing all models
for model, name in zip([nb_model, lr_model, rf_model], ['Naive Bayes', 'Logistic Regression', 'Random Forest']):
    print(f"\n{name} Results on Test Data:")
    y_pred = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
# Evaluating optimized models
for model, name in zip([best_nb_model, best_lr_model, best_rf_model], ['Optimized Naive Bayes', 'Optimized Logistic Regression', 'Optimized Random Forest']):
    print(f"\n{name} Results on Test Data:")
    y_pred = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
    print("Missing values found in the training data!")
    X_train = np.nan_to_num(X_train)  # Replace NaNs with 0 or a specific value
    y_train = np.nan_to_num(y_train)  # Same for the target variable

# Ensure the target variable is binary and numeric
y_train = y_train.astype(int)

# Initialize the SVM classifier
svm_model = SVC(random_state=42, kernel='linear', class_weight='balanced')

# RandomizedSearchCV for hyperparameter tuning with 2 iterations
random_params = {
    'C': np.logspace(-3, 3, 4),  # Fewer values to try
    'gamma': ['scale', 'auto'],
    'kernel': ['linear']
}

random_search = RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=random_params,
    scoring='f1',
    n_iter=2,  # Only 2 iterations for faster tuning
    cv=3,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Get the best model from RandomizedSearchCV
best_random_model = random_search.best_estimator_
print("\nBest Parameters (Random Search):", random_search.best_params_)

# Predict using the best model
y_pred_test = best_random_model.predict(X_test)

# Evaluate the model
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
    plt.yticks([0, 1], ["Non-Fraud", "Fraud"])
    plt.show()

evaluate_model(y_test, y_pred_test, "SVM with Random Search") 
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
