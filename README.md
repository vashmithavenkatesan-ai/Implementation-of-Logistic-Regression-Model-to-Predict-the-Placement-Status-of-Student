# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the student placement dataset.
2. Split the data into input features and target, then divide it into training and testing sets.
3. Train the Logistic Regression model using the training data after scaling the features.
4. Predict the placement status and evaluate the model using accuracy and confusion matrix.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VASHMITHA V
RegisterNumber:  212225240180
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create sample dataset
data = pd.DataFrame({
    'cgpa': [6.8, 5.9, 5.3, 7.4, 5.8, 7.1, 6.5, 8.2, 5.0, 7.8],
    'iq': [123, 106, 121, 132, 142, 115, 98, 140, 110, 128],
    'placement': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
})

print("Dataset Preview:")
print(data.head())

# Features and target
X = data[['cgpa', 'iq']]
y = data['placement']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# New student prediction (use DataFrame to avoid warning)
new_student = pd.DataFrame({
    'cgpa': [7.5],
    'iq': [120]
})

new_student = scaler.transform(new_student)
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nThe student is Placed")
else:
    print("\nThe student is Not Placed")

```

## Output:
<img width="1920" height="1080" alt="Screenshot (134)" src="https://github.com/user-attachments/assets/c15730f7-48f2-48e7-bb64-a6c599a6bd2c" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
