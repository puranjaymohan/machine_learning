import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def gradientd(X,Y,theta,a,i):
    for j in range(i):
        
        for k in range(X.shape[0]):
            theta[0,0]=theta[0,0]-(a*(X[k,0]*(X[k,:].dot(theta.T) -Y[k])))
            theta[0,1]=theta[0,1]-(a*(X[k,1]*(X[k,:].dot(theta.T) -Y[k])))
        
        
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
a=0.00000000005
i=1000

thetaf=gradientd(X,Y,theta,a,i)
print(thetaf)
cost=(1/(2*X.shape[0]))*np.sum((X.dot(thetaf.T) - Y)**2)
print(cost)
plt.xlabel('Area of House')
plt.ylabel('Price of House')
xx=np.arange(500,4500)
yy=xx*thetaf[0,0]+thetaf[0,1]
plt.plot(xx,yy)

plt.show()

##########Using Batch gradient descent##################
##theta at minima [[ 162.01523275 7778.81131956]]
##cost at minima 2327752580.2484665



###########Using Stochastic Gradient descent, this script##########
##theta at minima [[1.65360934e+02 9.39876086e-02]]
##minimum cost 2397865970.6467686







