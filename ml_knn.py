# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 00:08:45 2018

@author: BHANU
"""

import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier

headers = ['age','year','nodes']

raw = open('haberman.data.txt')

sur_stat = []
features = []


#Preparing the data
for line in raw:
    i=0
    var = []
    for y in line.split(','):
        if (i==3):
            sur_stat.append([int(y)])
        else:
            var.append(int(y))
        i+=1
    features.append(var)

df_x = pd.DataFrame.from_records(features, columns = headers)
df_y = pd.DataFrame.from_records(sur_stat, columns = ['label'])


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        df_x,df_y,test_size=0.2,random_state = 5)

knn = KNeighborsClassifier()

knn.fit(X_train,Y_train['label'].tolist())

y_pred = knn.predict(X_test)

correct = 0
for i in range(len(y_pred)):
    if(y_pred[i]==Y_test['label'].tolist()[i]):
        correct+=1

accuracy = correct/len(y_pred)

print('Accuracy = %s' % accuracy)
