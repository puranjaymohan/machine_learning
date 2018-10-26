import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(z):
    return 1/(1+np.exp(-z))

def minima(X,Y,theta,a,it):
    for ab in range(it):
        for i in range(X.shape[0]):
            theta[0,0] = theta[0,0] + a*X[i,0]*(Y[i,0]-sigmoid(X[i].dot(theta.T)))
            theta[0,1] = theta[0,1] + a*X[i,1]*(Y[i,0]-sigmoid(X[i].dot(theta.T)))
    return theta

df=pd.read_csv('data.txt')

X1=np.array([df['a']])
X1=X1.T

Y=np.array([df['b']])
Y=Y.T
one=np.ones((X1.shape[0],1))
X=np.concatenate((X1,one),axis=1)
theta=np.array([[0,0]],dtype=float)
a=0.1
it=10000

theta=minima(X,Y,theta,a,it)
print(theta)
xx=np.linspace(-15,15,num=1000)
yy=sigmoid(theta[0,0]*xx+theta[0,1])

xxx=np.linspace(-0.1,1.1,num=3)
yyy= (-theta[0,1]/theta[0,0])
yyy=[yyy] *3
plt.plot(xx,yy)
plt.plot(yyy,xxx)
plt.legend(['Sigmoid Curve','Decision Boundary'])
plt.scatter(X1[0:22],Y[0:22],c='r',marker='o')
plt.scatter(X1[22:],Y[22:],c='b',marker='o')
plt.show()

