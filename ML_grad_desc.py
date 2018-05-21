#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:29:14 2018

@author: bh387886
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt



def Grad_desc(X, y, m=0
              ,iters=10000, alpha=0.001):
     N = len(y)
     print('Iterations: %s, Learning Rate: %s' % (iters,alpha))
     c = (float(random.randint(0,30))/100)
     for i in range(iters):
         m_grad = 0
         c_grad = 0
         for j in range(N):
             m_grad += -(y[j]-(m*X[j]+c))*X[j]/N
             c_grad += -(y[j]-(m*X[j]+c))/N
         m = m - alpha*m_grad
         c = c - alpha*c_grad
     cost = 0
     for j in range(N):
         cost += (y[j]-(m*X[j]+c))**2
     return [m, c, cost/(2*N)]
 



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

x_org = df_x['horsepower'] #change this if you want another dependent var
y_org = df_y['mpg']

#Feature Scaling (Reducing the )
df_new1 = (x_org-x_org.min())/(x_org.max()-x_org.min())
df_y1 = (y_org - y_org.min())/(y_org.max() - y_org.min())


ans = Grad_desc(df_new1.tolist(),df_y1.tolist())

y_ans = ans[0]*df_new1 + ans[1]

#Restoring the scaled features
y_ans = y_ans*(y_org.max() - y_org.min()) + y_org.min()
ans[1] = ans[1]*(y_org.max() - y_org.min()) + y_org.min()
ans[0] = (y_ans[1]-y_ans[0])/(x_org[1]-x_org[0])
   
   
print('ERROR = %s' % ans[2])
print('Equation of best-fit-line: y=%sx + %s' % (ans[0],ans[1]))


 


plt.scatter(x_org, df_y,  color='black')
plt.plot(x_org, y_ans, color='blue', linewidth=3)