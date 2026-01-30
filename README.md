# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MOHANAPRABHA S
RegisterNumber: 212224040197
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test ,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title ("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test ,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='Red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE= ',mae)
```

## Output:
## TRAINING SET:
<img width="700" height="563" alt="image" src="https://github.com/user-attachments/assets/f36be434-6056-42d2-a2ee-9dee138573be" />

## TESTING SET
<img width="699" height="569" alt="image" src="https://github.com/user-attachments/assets/fe4535f2-471e-4b7b-acfd-1f7aae77d525" />

## DATASET:

<img width="260" height="567" alt="image" src="https://github.com/user-attachments/assets/6db143d0-fe22-49ab-8943-fefb0f8e1395" />

## HEAD VALUES:

<img width="278" height="130" alt="image" src="https://github.com/user-attachments/assets/8a4854c2-9bed-4fc8-8dbb-038c19a2814a" />

## TAIL VALUES:
<img width="259" height="128" alt="image" src="https://github.com/user-attachments/assets/1e1ac132-3bc2-4d25-b69f-ca8bf26a2f67" />

## X AND Y VALUES:

<img width="876" height="586" alt="image" src="https://github.com/user-attachments/assets/8a6fcb54-e84d-4aff-b495-acebb5f525a8" />

## PREDICTION VALUES OF X AND Y VALUES:

<img width="730" height="65" alt="image" src="https://github.com/user-attachments/assets/e771cc88-3c15-4f3a-ae20-317053723a03" />

## MSE,MAE,AND RMSE:

<img width="274" height="49" alt="image" src="https://github.com/user-attachments/assets/d4213bb6-3a55-464f-a175-db7901c4b4fe" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
