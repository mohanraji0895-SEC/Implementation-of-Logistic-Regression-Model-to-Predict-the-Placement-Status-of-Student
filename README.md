# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Load the placement dataset and create a copy, dropping the unwanted columns 'sl_no' and 'salary'
3. Check the data for missing values and duplicate rows
4. Encode the categorical columns using LabelEncoder
5. Separate the data into features (X) and target (y)
6. Split the data into training and testing sets
7. Create and train the Logistic Regression model using the training set
8. Predict the placement status on the test set and evaluate accuracy and classification report
9. Predict the placement status for a new student
10. Stop

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MOHANRAJI D
RegisterNumber: 212225060164



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Placement_Data.csv")
print(data.head())

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)
print(data1.head())

print(data1.isnull().any())
print(data1.duplicated().sum())

cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()
for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])
print(data1.head())

X = data1.iloc[:, :-1]
y = data1["status"]
print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(classification_report(y_test, y_pred))

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
new_prediction = lr.predict(new_student)
print(new_prediction[0])

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
