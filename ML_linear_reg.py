#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:27:11 2018

@author: bh387886
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

headers = ['cylinders','displacement','horsepower','weight'
           ,'acceleration','model year','origin']



raw = open('auto-mpg.data.txt')

mpg = [] #Dependent variable : mpg
features = [] #Independent variables: rest of them except car name


#Preparing the data
for line in raw:
    i=0
    var = []
    for y in line.split():
        if (i==0):
            mpg.append([float(y)])
        if (i<8 and i>0):
            if(y=='?'): # Handaling missing values for horsepower 
                y = 0   # (Setting them to 0)
            if(i==4):
                var.append(float(y.replace('.','')))
            else:
                var.append(float(y))
        i+=1
    features.append(var)
    
 
df_x = pd.DataFrame.from_records(features, columns = headers)
df_y = pd.DataFrame.from_records(mpg, columns = ['mpg'])

#replacing 0(missing) horsepower with average horsepower
avg_bhp = (np.average(df_x['horsepower'])*len(df_x))/(len(df_x)-6)
df_x['horsepower'] = df_x['horsepower'].replace(0,avg_bhp)

lm = LinearRegression()
lm1 = LinearRegression()

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        df_x,df_y,test_size=0.2,random_state = 5)



######################## USING ALL INDEPENDENT VARS ###########################

lm.fit(X_train, Y_train)

pred_test = lm.predict(X_test)

print('coefficients :(all vars)')
print(lm.coef_)

print('Intercept :(all vars)')
print(lm.intercept_)

print('Accuracy Score(all vars): %s' % lm.score(X_test,Y_test))

residues = (Y_test - pred_test)

#print('Residues: ')
#print((residues))

###############################################################################

print('####################################################')

##################### USING ALL ONE OF THE VAR(horse power) ###################

df_new1 = df_x['horsepower']

df_new = df_new1 [:, np.newaxis]

X_train1, X_test1, Y_train1, Y_test1 = sklearn.model_selection.train_test_split(
        df_new,df_y,test_size=0.2,random_state = 5)


lm1.fit(X_train1, Y_train1)

pred_test1 = lm1.predict(X_test1)

print('coefficients :(single var)')
print(lm1.coef_)

print('Intercept :(single var)')
print(lm1.intercept_)

print('Accuracy Score(single vars): %s' % lm1.score(X_test1,Y_test1))

residues = (Y_test1 - pred_test1)

#print('Residues: ')
#print((residues))


plt.scatter(X_test1, Y_test1,  color='black')
plt.plot(X_test1, pred_test1, color='blue', linewidth=3)

###############################################################################

