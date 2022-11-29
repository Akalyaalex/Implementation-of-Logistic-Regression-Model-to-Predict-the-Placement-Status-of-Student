# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset

2.Check for null and duplicate values

3.Assign x and y values

4.Split the data into training and testing data

5.Import logistic regression and fit the training data

6.Predict y value

7.Calculate accuracy and confusion matrix

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Akalya A
RegisterNumber: 212220220002
#LOGISTIC REGRESSION
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion
*/
```

## Output:
![4 1](https://user-images.githubusercontent.com/114275126/204471978-1ed96931-1b7a-4e61-92f2-cc3996d72451.PNG)

![4 2](https://user-images.githubusercontent.com/114275126/204472038-3d17aa30-0d9a-4be2-b590-b7c840ba1d72.PNG)

![4 3](https://user-images.githubusercontent.com/114275126/204472134-b251be78-170c-4931-b813-7da093972e03.PNG)

![4 4](https://user-images.githubusercontent.com/114275126/204472191-99746d4b-3427-43b6-92e3-90d62b6841da.PNG)

![4 5](https://user-images.githubusercontent.com/114275126/204472270-b18a5e6d-0700-4bcb-b096-d1b24f5551bc.PNG)

![4 6](https://user-images.githubusercontent.com/114275126/204472326-9951bc24-52c7-4549-8904-27cb6b0a2a6e.PNG)

![4 7](https://user-images.githubusercontent.com/114275126/204472384-0572d90a-63cf-4762-903f-c766c083a77b.PNG)

![4 8](https://user-images.githubusercontent.com/114275126/204472427-1995b95e-a3f3-4dce-8c81-4fce3aadd6d1.PNG)

![4 9](https://user-images.githubusercontent.com/114275126/204472532-78e4ccd9-affe-4f09-9257-b138f44e4903.PNG)

![4 10](https://user-images.githubusercontent.com/114275126/204472580-2358891c-6b12-4705-a869-f92a271297ef.PNG)

![4 11](https://user-images.githubusercontent.com/114275126/204472639-debca8f3-bb3a-4606-86e1-02df5fae1e7a.PNG)

![4 12](https://user-images.githubusercontent.com/114275126/204472652-31c389d6-4773-42e8-986e-211d6e7d93ee.PNG)

![4 13](https://user-images.githubusercontent.com/114275126/204472762-2066942f-686e-476f-b9bc-2356435b82f0.PNG)

![4 14](https://user-images.githubusercontent.com/114275126/204473217-8eb4ff02-6a40-4b80-a9f5-22e71776c5e6.PNG)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
