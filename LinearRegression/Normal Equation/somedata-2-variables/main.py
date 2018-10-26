import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random





######Cost Function#########
def getcost(X,Y,theta):
    r=(1/(2*X.shape[0]))*np.sum(np.square(Y-X.dot(theta.T)))
    return r


##### The function to implement gradient descent algorithm #####
def normalit(X,Y,theta):
    theta = ((np.linalg.inv(X.T.dot(X))).dot(X.T)).dot(Y)
    return theta


def accu(X,Y,theta):
    r=sqrt((np.sum((Y-X.dot(theta.T))**2/(X.shape[0])))-(((np.sum(Y-X.dot(theta.T)))**2)/(X.shape[0])**2))
    return r





#setting up the data
d=pd.read_csv('somedata.csv')
x=d['Price (Older)']
y=d['Price (New)']
x1=d['Price (Modern)']


#Creating Matrices from data for computations
Y=np.array([y])    #row vector of new price column from data
Y=Y.T    #Converting it to a column vector for the sake of maths
one=np.ones((len(x),1))   # creating a column verctor of "1" with no of rows 
X1=np.array([x1])
X1=X1.T

X=np.array([x])      #row vector of old price column from the data
X=X.T                #Making it a column vector for the sake of maths
X=np.concatenate((X,X1), axis=1) # making a matrix with two columns by joing X and one
X=np.concatenate((X,one), axis=1)
theta=np.array([[0,0,0]],dtype=float) #theta it is in the form [theta-1,theta-not]

result=normalit(X,Y,theta) #getting the values of theta for which cost is minimum
result=result.T
print(f"Result - {result}")
print(f"Cost at Result - {getcost(X,Y,result)}")
print(f"RMSE - {accu(X,Y,result)}")

xx=np.linspace(0,200000,50)
ll=np.linspace(0,200000,50)

xx,ll =np.meshgrid(xx,ll)
yy=xx*result[0,0] + ll*result[0,1] + result[0,2]


#----Plotting-------
fig = pyplot.figure()
ax = Axes3D(fig)

sequence_containing_x_vals = x
sequence_containing_y_vals = x1
sequence_containing_z_vals = y

random.shuffle(sequence_containing_x_vals)
random.shuffle(sequence_containing_y_vals)
random.shuffle(sequence_containing_z_vals)

ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
ax.plot_surface(xx,ll,yy)
pyplot.show()
