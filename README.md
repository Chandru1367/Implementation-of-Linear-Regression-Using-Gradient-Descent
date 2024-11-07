# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries such as NumPy, Pandas, Matplotlib, and metrics from sklearn.
2.Load the dataset into a Pandas DataFrame and preview it using head() and tail().
3.Extract the independent variable X and dependent variable Y from the dataset.
4.Initialize the slope m and intercept c to zero. Set the learning rate L and define the number of epochs.
5.Plot the error against the number of epochs to visualize the convergence.
6.Display the final values of m and c, and the error plot.
## Program:
```python
Program to implement the linear regression using gradient descent.
Developed by: M.Chandru
RegisterNumber: 24900224 
```
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01, num_iters=1000):
     X=np.c_[np.ones(len(X1)), X1]
     theta=np.zeros (X.shape[1]).reshape(-1,1)
     for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
     return theta
    
data=pd.read_csv("50_Startups.csv",header=None)
X=(data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot (np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print (data.head)
print(f"Predicted value: {pre}")
```

## Output:
![Screenshot 2024-11-07 091140](https://github.com/user-attachments/assets/9b2b531a-0701-47e0-9fd6-f555616ab55f)
![Screenshot 2024-11-07 091155](https://github.com/user-attachments/assets/4e1a2606-ebbb-412b-bd8e-a2e14193e5d9)
![Screenshot 2024-11-07 091202](https://github.com/user-attachments/assets/b91ab9bc-45f1-44fc-828e-e874d163ffdb)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
