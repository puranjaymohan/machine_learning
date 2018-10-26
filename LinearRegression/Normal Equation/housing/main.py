import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def normal(X,Y,theta):
    theta=(np.linalg.inv(X.T.dot(X))).dot(X.T).dot(Y)
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

thetaf=normal(X,Y,theta)
thetaf=thetaf.T
print(thetaf)
cost=(1/(2*X.shape[0]))*np.sum((X.dot(thetaf.T) - Y)**2)
print(cost)
plt.xlabel('Area of House')
plt.ylabel('Price of House')
xx=np.arange(500,4500)
yy=xx*thetaf[0,0]+thetaf[0,1]
plt.plot(xx,yy)

plt.show()

##theta at minima [[ 162.01523275 7778.81131956]]
##cost at minima 2327752580.2484665

