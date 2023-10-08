# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,svm
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score)
import pandas as pd
from sklearn.model_selection import train_test_split

#%% Performance evaluation function
def eval_perform(Y,Yhat):
    accu = accuracy_score(Y,Yhat)
    prec = precision_score(Y,Yhat,average='weighted')
    reca = recall_score(Y,Yhat,average='weighted')
    print('\n \t Accu \t Prec \t Reca\n Eval \t %0.3f \t %0.3f \t %0.3f'%(accu,prec,reca))


#%% Import data (EXAMPLE 2)
data = pd.read_csv('../Data/ex2data2.txt',header=None)
X = data.iloc[:,0:2]
Y = data.iloc[:,2]

#%% Data visualization
fig = plt.figure(figsize=(8,8))
indx = Y==1
plt.scatter(X[0][indx],X[1][indx],c='g',label='Class: +1')
plt.scatter(X[0][~indx],X[1][~indx],c='r',label='Class: -1')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()

plt.show()

#%% Prepare the datasets to optimize the hyperparameter desired
# The train, crosvalidation and tes subsets are created
X_train, X_, Y_train, Y_ = train_test_split(X, Y, test_size=0.3, random_state=10)
X_cv, X_test, Y_cv, Y_test = train_test_split(X_, Y_, test_size=0.5, random_state=10)

#%% Optimization of the degree of a polynomial kernel

C = 1
degree_values = np.arange(1,10) # range of values to test

accu_train = np.zeros(np.shape(degree_values))
accu_cv = np.zeros(np.shape(degree_values))
accu_test = np.zeros(np.shape(degree_values))

for k in range(len(degree_values)):
    modelo = svm.SVC(kernel='poly',degree=degree_values[k],C=C) #create the model
    modelo.fit(X_train,Y_train) # fiting the model with train dataset
    
    # metric evaluation
    accu_train[k] = accuracy_score(Y_train,modelo.predict(X_train))
    accu_cv[k] = accuracy_score(Y_cv,modelo.predict(X_cv))
    accu_test[k] = accuracy_score(Y_test,modelo.predict(X_test))

#%% View the performance of the train and cross-validation
fig = plt.figure(figsize=(8,4))
plt.plot(degree_values, accu_train,linewidth=4, markersize=12, c='b',marker='o', label='train')
plt.plot(degree_values, accu_cv,linewidth=4, markersize=12, c='r',marker='o', label='cv')
# plt.plot(degree_values, accu_test,linewidth=4, markersize=12, c='g',marker='o', label='test')
plt.xlabel('degree value')
plt.ylabel('MSE')
plt.title('Optimization with C=%0.2f'%(C))
plt.legend()

#%% Creating the SVC model optimizzed

modelo = svm.SVC(kernel='poly',degree=2,C=C)

modelo.fit(X_train,Y_train)


print('\n\nTraining Evaluation')
Yhat = modelo.predict(X_train)
eval_perform(Y_train,Yhat)

print('\n\nCross-Validation Evaluation')
Yhat = modelo.predict(X_cv)
eval_perform(Y_cv,Yhat)

print('\n\nTesting Evaluation')
Yhat = modelo.predict(X_test)
eval_perform(Y_test,Yhat)


#%% View the decision boundary
h = 0.01
xmin,xmax,ymin,ymax = X[0].min(),X[0].max(),X[1].min(),X[1].max()
xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))

Xnew = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()])

Z = modelo.predict(Xnew)
Z = Z.reshape(xx.shape)

vs = modelo.support_vectors_

indx = Y==1
fig = plt.figure(figsize=(8,8))
plt.scatter(X[0][indx],X[1][indx],c='g',label='Class: +1')
plt.scatter(X[0][~indx],X[1][~indx],c='r',label='Class: -1')
plt.contour(xx,yy,Z)
plt.scatter(vs[:,0],vs[:,1],s=60,marker='x',facecolors='k')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
# fig.savefig('../figures/fig2_svm_2d.png')
plt.show()
