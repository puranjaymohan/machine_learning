################

#NOTHING 
#NOW
################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
def sigmoid(x) :
    return 1/(1+np.exp(-x))

def loss(X,theta,Y) :
    scores = np.dot(X, theta.T)
    ll = np.sum( Y*scores - np.log(1 + np.exp(scores)) )
    return ll


def dx(X,i,Y,theta) :
    return X[:,i].dot(Y - X.dot(theta.T))

def ddx(X,i,Y,theta) :
    return 
    





df=pd.read_csv('data.txt')

Y=np.array([df['e']])
Y=Y.T

one=np.ones((Y.shape[0],1))
X1=np.array([df['b']])
X1=preprocessing.normalize(X1)
X1=X1.T


X2=np.array([df['a']])
X2=preprocessing.normalize(X2)
X2=X2.T

X=np.concatenate((X2,X1),axis=1)
X=np.concatenate((X,one),axis=1)



theta=np.array([[0,0,0]],dtype=float)
a=1.5
it=1500000


theta=maxima(X,Y,theta,a,it)

print(theta)

print(loss(X,theta,Y))


xx=np.linspace(0.05,0.115,3);
yy=(-theta[0,2]-theta[0,0]*xx)/theta[0,1]

plt.figure(figsize=(12,8))
plt.scatter(X[0:49, 0], X[0:49, 1])

plt.scatter(X[50:, 0], X[50:, 1])

plt.plot(xx,yy,'g')
plt.show()
