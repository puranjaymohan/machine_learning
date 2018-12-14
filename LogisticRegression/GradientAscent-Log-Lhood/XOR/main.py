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
    return ll


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


a=1
it=500000
theta=np.array([[0,0,0,0]])
print(theta.shape,X.shape,Y.shape)
theta=maxima(X,Y,theta,a,it)
print(theta)

xx, yy =np.meshgrid(np.arange(-1,2,0.5), np.arange(-1,2,0.5))


z=((-theta[0,0]*xx-theta[0,1]*yy-theta[0,3])/theta[0,2])



###########


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[0:2,0],X[0:2,1],X[0:2,2],c='r',marker='x')

ax.scatter(X[2:,0],X[2:,1],X[2:,2],c='b',marker='o')

ax.plot_surface(xx,yy,z,alpha=0.9,cmap=cm.summer)
ax.set_zlim(0,1.2)

plt.show()






