# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 02:00:36 2017

@author: Saurabh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from matplotlib import style
style.use('ggplot')
df = pd.read_excel("D:\Python_practice\Baseball.xls")

print("<------------------df dataframe------------------->")
print(df.head())
print()
print()

x1 = np.array(df.X2)
x2 = np.array(df.X3)
x3 = np.array(df.X4)
x4 = np.array(df.X5)
x5 = np.array(df.X6)
y  = np.array(df.X1)


#Feature Scaling
Scaled_X1 = preprocessing.scale(x1)
Scaled_X2 = preprocessing.scale(x2)
Scaled_X3 = preprocessing.scale(x3)
Scaled_X4 = preprocessing.scale(x4)
Scaled_X5 = preprocessing.scale(x5)
z = pd.DataFrame()
z['x1'] = Scaled_X1
z['x2'] = Scaled_X2
z['x3'] = Scaled_X3
z['x4'] = Scaled_X4
z['x5'] = Scaled_X5
print("<------------------z(training) dataframe------------------->")
print(z.head())
print()
Y = preprocessing.scale(y)

#Taking 75% of Scaled_X and Scaled_Y for training and rest 25% for testing purpose
X_train , X_test ,  Y_train , Y_test = cross_validation.train_test_split(z,Y,test_size=0.25)

#Shape of X_train should be array(number_of_sample,number_of_feature) 
#If you have only one feature and shape of array is not like (number_of_sample,1) , then reshape --> X_train.reshape(len(X_train),1)

#Now here comes linear regression
clf = LinearRegression()
clf.fit(X_train,Y_train)  #Training




accuracy = clf.score(X_test,Y_test)
print("Accuracy:",accuracy*100,"%")

#visualisation for every feature...replace feature X-axis 
#Accuracy can be visualize by watching the difference between red and white dots.
plt.scatter(X_test.x1,Y_test,color='black')
plt.scatter(X_test.x1,clf.predict(X_test),color='red')


