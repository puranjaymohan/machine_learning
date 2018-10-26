import numpy as np
from matplotlib import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *
import random
import scipy.interpolate as interp


def getcost(X,Y,theta):                                                     
     r=(1/(2*X.shape[0]))*np.sum(np.square(Y-X.dot(theta.T)))
     return r



def gradientd(X,Y,theta,a,i):
    thetas0=list()
    thetas1=list()
    
    costs=list()
    for j in range(i):
        theta[0,0]=theta[0,0]-((a/X.shape[0])*(X[:,0].T.dot(X.dot(theta.T) -Y)))
        theta[0,1]=theta[0,1]-((a/X.shape[0])*(X[:,1].T.dot(X.dot(theta.T) -Y)))
        #print(theta)
    return theta



d=pd.read_csv('data.txt')
x=d['area']
y=d['price']
plt.scatter(x,y)
X=np.array([x])
X.shape=(len(x),1)
Y=np.array([y])
Y.shape=(len(y),1)
one=np.ones((X.shape[0],1))
X=np.concatenate((X,one),axis=1)

theta=np.array([[0,0]],dtype=float)
a=0.000000432
i=2000000

thetaf=gradientd(X,Y,theta,a,i)
print(thetaf)
cost=(1/(2*X.shape[0]))*np.sum((X.dot(thetaf.T) - Y)**2)
print(cost)




plt.xlabel('Area')
plt.ylabel('Price')

xx=np.arange(500,4500)
yy=xx*thetaf[0,0]+thetaf[0,1]
plt.plot(xx,yy)
################
plt.show()



##theta at minima [[ 162.01523275 7778.81131956]]
##cost at minima 2327752580.2484665

