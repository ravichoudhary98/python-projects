import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sb
import numpy as np
import matplotlib.pyplot as py

df = pd.read_csv('/home/ravi/Documents/weight-height.csv')
df['Gender'] = df['Gender'].replace('Male', 0) 
df['Gender'] = df['Gender'].replace('Female', 1)

X = df[['Weight', 'Height']]
Y = df[['Gender']]

X_train, X_test, y_train, y_test = train_test_split(X, Y)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

sb.distplot(y_test - predictions, axlabel="Test - Prediction")
py.show()

myvals = np.array([177.992066, 69.868511]).reshape(1, -1)
print(model.predict(myvals)) 
