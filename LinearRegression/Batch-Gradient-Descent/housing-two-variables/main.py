import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random










def gradientd(X,Y,theta,a,i):
    for j in range(i):
        theta[0,0]=theta[0,0]-((a/X.shape[0])*(X[:,0].T.dot(X.dot(theta.T) -Y)))
        theta[0,1]=theta[0,1]-((a/X.shape[0])*(X[:,1].T.dot(X.dot(theta.T) -Y)))
        theta[0,2]=theta[0,2]-((a/X.shape[0])*(X[:,2].T.dot(X.dot(theta.T) -Y)))
        #print(theta)
    return theta



d=pd.read_csv('data.txt')
x=d['area']
y=d['price']
x2=d['bed']
#plt.scatter(x,x2,y)
X=np.array([x])
X.shape=(len(x),1)

X1=np.array([x2])
X1.shape=(len(x2),1)

Y=np.array([y])
Y.shape=(len(y),1)
one=np.ones((X.shape[0],1))
X=np.concatenate((X,X1), axis=1)
X=np.concatenate((X,one),axis=1)

theta=np.array([[0,0,0]],dtype=float)

a=0.000000432
i=2000000

thetaf=gradientd(X,Y,theta,a,i)
print(thetaf)
cost=(1/(2*X.shape[0]))*np.sum((X.dot(thetaf.T) - Y)**2)
print(cost)
#plt.xlabel('Area of House')
#plt.ylabel('Price of House')
xx=np.linspace(500,4500,30)

l1=np.linspace(0,6,30)

#print(xx)
#print(xx1)
#print(thetaf[0,0],thetaf[0,1],thetaf[0,2])

xx, l1 =np.meshgrid(xx,l1)
yy= (xx.dot(thetaf[0,0]))+l1*thetaf[0,1] + thetaf[0,2]

#print(yy)
#plt.plot(xx,yy)

#plt.show()

##theta at minima [[ 162.01523275 7778.81131956]]
##cost at minima 2327752580.2484665



fig = pyplot.figure()
ax = Axes3D(fig)

sequence_containing_x_vals = x
sequence_containing_y_vals = x2
sequence_containing_z_vals = y

random.shuffle(sequence_containing_x_vals)
random.shuffle(sequence_containing_y_vals)
random.shuffle(sequence_containing_z_vals)

ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
ax.plot_surface(xx,l1,yy)
pyplot.show()
