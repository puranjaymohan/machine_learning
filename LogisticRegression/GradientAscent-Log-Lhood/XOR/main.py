import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import *
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(z):
    return 1/(1+np.exp(-z))


def loss(X,theta,Y) :
    scores = np.dot(X, theta.T)
    ll = np.sum( Y.reshape(4,1)*scores - np.log(1 + np.exp(scores)) )
    return -ll


def maxima(X,Y,theta,a,it):
    for lll in range(it):
        
        theta = theta + ((a)*X.T.dot(Y.reshape(4,1) - sigmoid(X.dot(theta.T)))).T
        if lll % 10000 == 0 :
            print(loss(X,theta,Y))
    return theta




data = pd.read_csv('data.csv')
print(data.head())
Y=np.array(data['o'].replace(-1,0))
X=np.array(data[['a','b']])
X1=np.array([0,1,0,0]).reshape(4,1)

X=np.concatenate((X,X1),axis=1)
ones=np.ones((4,1))
X=np.concatenate((X,ones),axis=1)
print(X)


a=9
it=800000
theta=np.array([[0,0,0,0]])
print(theta.shape,X.shape,Y.shape)
theta=maxima(X,Y,theta,a,it)
print(theta)

xx, yy =np.meshgrid(np.arange(-0.2,1.2,0.01), np.arange(-0.2,1.2,0.01))


z=+theta[0,0]*xx.ravel()+theta[0,1]*yy.ravel()+theta[0,3]+theta[0,2]*(np.multiply(xx,yy).ravel())


###########


fig = plt.figure()

plt.scatter(xx.ravel(),yy.ravel(),c=z)

plt.scatter(X[0:2,0],X[0:2,1],c='r',marker='x',s=40)

plt.scatter(X[2:,0],X[2:,1],c='b',marker='o',s=40)




plt.show()






