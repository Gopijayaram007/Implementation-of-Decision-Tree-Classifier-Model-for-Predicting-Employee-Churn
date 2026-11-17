# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import necessary libraries such as pandas, sklearn.model_selection, sklearn.tree, sklearn.preprocessing, and sklearn.metrics.
2. Load Dataset: Load the Employee dataset (Employee.csv) using pandas and inspect its structure, data types, and null values.
3. Data Preprocessing: Clean column names (if needed) and encode categorical variables like Departments and salary into numerical format using LabelEncoder.
4. Define Features and Target: Separate the independent feature variables ($X$) from the target variable (left, $Y$).
5. Split Data: Split the dataset into training and testing sets (e.g., 80% train, 20% test) using train_test_split.
6. Train Model: Initialize the DecisionTreeClassifier and fit it on the training data.
7. Evaluate Model: Predict churn on the test dataset and evaluate performance using metrics such as accuracy score, confusion matrix, and classification report.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('Employee.csv')

df.columns = df.columns.str.strip()

print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

le_salary = LabelEncoder()
df['salary'] = le_salary.fit_transform(df['salary'])

le_dept = LabelEncoder()
df['Departments'] = le_dept.fit_transform(df['Departments'])
X = df.drop('left', axis=1)
y = df['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

print("\n--- Model Evaluation ---")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Output:

<img width="667" height="686" alt="image" src="https://github.com/user-attachments/assets/ab2ee63c-f0bf-455c-86c1-83451ec4ae46" />
<img width="497" height="257" alt="image" src="https://github.com/user-attachments/assets/a0b40737-1daf-4013-a1af-edc26ff83d6e" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
