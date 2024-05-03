import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import SVC # Change this to the model you want to use
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the embeddings and labels from the CSV file
train_embeddings_file = r'Filepath to the train embeddings file'
eval_embeddings_file = r'Filepath to the eval embeddings file'

train_df = pd.read_csv(train_embeddings_file)
eval_df = pd.read_csv(eval_embeddings_file)

# Split the data into X and y
X_train = train_df.iloc[:, :-1].values # Select all columns except the last one
y_train = train_df.iloc[:, -1].values  # Select the last column(Labels)

X_eval = eval_df.iloc[:, :-1].values
y_eval = eval_df.iloc[:, -1].values

# Parameter grid for the grid search based on the ML model
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Specifies the kernel type to be used in the algorithm
    'degree': [3, 4, 5],  # Degree of the polynomial kernel function ('poly')
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
}

# Loading the model
svm = SVC(probability=True)  

# Instantiating the grid search model
model = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

# Fit the grid search to the data
model.fit(X_train, y_train)

# Best parameters found
print("Best parameters:", model.best_params_)

best_svm_model = model.best_estimator_

# Evaluate the model on the eval data
y_pred = best_svm_model.predict(X_eval)

# Evaluation metrics
accuracy = accuracy_score(y_eval, y_pred)
f1 = f1_score(y_eval, y_pred)
recall = recall_score(y_eval, y_pred)
precision = precision_score(y_eval, y_pred)

print("SVM Model Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")

print('Confusion Matrix:')
print(confusion_matrix(y_eval, y_pred))
print('Classification Report:')
print(classification_report(y_eval, y_pred))

# Loading the test data
test_embeddings_file = r'Filepath to the test embeddings file'
test_df = pd.read_csv(test_embeddings_file)

# Split the data into X and y
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Evaluate the model on the TEST data
y_pred = best_svm_model.predict(X_test) 

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# Print the evaluation metrics
print("SVM Model Test Metrics:")
print(f"Test Accuracy: {accuracy}")
print(f"Test F1 Score: {f1}")
print(f"Test Recall: {recall}")
print(f"Test Precision: {precision}")