import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
######Cost Function#########
def getcost(X,Y,theta):
    r=(1/(2*X.shape[0]))*np.sum(np.square(Y-X.dot(theta.T)))
    return r


##### The function to implement gradient descent algorithm #####
def changetheta(X,Y,theta,a,itrations):
    for i in range(itrations):
        theta[0,0] = theta[0,0] - (a/X.shape[0])*(X[:,0].T.dot(X.dot(theta.T) - Y))
        theta[0,1] = theta[0,1] - (a/X.shape[0])*(X[:,1].T.dot(X.dot(theta.T) - Y))
        ###just to see the chang in theta each time
        #print(theta)  
        
    return theta


def accu(X,Y,theta):
    r=sqrt((np.sum((Y-X.dot(theta.T))**2/(X.shape[0])))-(((np.sum(Y-X.dot(theta.T)))**2)/(X.shape[0])**2))
    return r





#setting up the data
d=pd.read_csv('somedata.csv')
x=d['Price (Older)']
y=d['Price (New)']



#Creating Matrices from data for computations
Y=np.array([y])    #row vector of new price column from data
Y=Y.T    #Converting it to a column vector for the sake of maths
one=np.ones((len(x),1))   # creating a column verctor of "1" with no of rows same as x
a=0.000000000003     #learning rate which fits best after tweaking
itrations=1000000    # i love 1 million
X=np.array([x])      #row vector of old price column from the data
X=X.T                #Making it a column vector for the sake of maths
X=np.concatenate((X,one), axis=1) # making a matrix with two columns by joing X and one
theta=np.array([[0,0]],dtype=float) #theta it is in the form [theta-1,theta-not]

result=changetheta(X,Y,theta,a,itrations) #getting the values of theta for which cost is minimum
print(f"Result - {result}")
print(f"Cost at Result - {getcost(X,Y,result)}")
print(f"RMSE - {accu(X,Y,result)}")

#----Plotting-------
plt.scatter(x,y) # scattering the real data
xx=np.arange(0,200000)
yy=result[0,0]*xx+result[0,1]
plt.plot(xx,yy)  # ploting the prediction on the same graph
plt.show()
