import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
####Cost Function, Just for the sake of trying####
def getcost(X,Y,theta):
    cost=0
    for i in range(X.shape[0]) :
        cost = cost + (X[i].dot(theta.T) - Y[i])**2
    cost = cost/(2*X.shape[0])
    return cost


#def getcost(X,Y,theta):
#    r=(1/(2*X.shape[0]))*np.sum(np.square(Y-X.dot(theta.T)))
#    return r

##################################################

def changetheta(X,Y,theta,a,itrations):
    for i in range(itrations):
        theta[0,0] = theta[0,0] - (a/X.shape[0])*(X[:,0].T.dot(X.dot(theta.T) - Y))
        theta[0,1] = theta[0,1] - (a/X.shape[0])*(X[:,1].T.dot(X.dot(theta.T) - Y))
        ###just to see the changin theta each time
       # print(theta)  
        
    return theta

############accuracy#########
def accu(X,Y,theta):
    r=sqrt((np.sum((Y-X.dot(theta.T))**2/(X.shape[0])))-(((np.sum(Y-X.dot(theta.T)))**2)/(X.shape[0])**2))
    return r


#setting up the data
d=pd.read_excel('boston.xls')
#print(d.head())

x=d['LSTAT']
#x=d['CRIM']
#x=d['TAX'] useless
#x=d['AGE'] useless
#x=d['DIS']
#x=d['RAD'] 
#x=d['PT'] # change stuff for this down in plotting and learning rate
y=d['MV']




Y=np.array([y])
Y=Y.T
one=np.ones((len(x),1))
a=0.001 #for LSTAT CRIM DIS RAD
#a=0.0007 #for PT

itrations=1000000
X=np.array([x])
X=X.T
X=np.concatenate((X,one), axis=1)
theta=np.array([[0,0]],dtype=float)


result=changetheta(X,Y,theta,a,itrations)
print(f"Result - {result}")
print(f"Cost at Result - {getcost(X,Y,result)}")
print(f"RMSE - {accu(X,Y,result)}")
#----Plotting-------
plt.scatter(x,y)

xx=np.arange(0,50)
#xx=np.arange(10,30,0.1) #for PT

yy=result[0,0]*xx+result[0,1]

plt.plot(xx,yy)
plt.show()
