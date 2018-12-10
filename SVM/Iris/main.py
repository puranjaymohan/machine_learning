import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import preprocessing
import matplotlib.patches as mpatches

def loss(X,Y,W):
    
    total_error = 0
    for i, x in enumerate(X):
        if (np.dot(X[i], W)*Y[i]) <= 0:
            total_error += (np.dot(X[i], W)*Y[i])
                
    return total_error

def fit(X,Y,w,a,i):
    
    errors=[]
    #training part, gradient descent part
    for epoch in range(1,i):
        
        for i, x in enumerate(X):
            #misclassification
            if (Y[i]*np.dot(X[i], w)) < 1:
                #misclassified update for ours weights
                w = w + a * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                
            else:
                #correct classification, update our weights
                w = w + a * (-2  *(1/epoch)* w)
        print(loss(X,Y,W))
        if epoch%100 == 0:
            errors.append([loss(X,Y,W),epoch])
        if epoch % 10000 == 0:
            print(epoch)
        
    return w,errors
    


df=pd.read_csv('data.txt')



y=np.array(df['e'], dtype='float64')
y11=np.array(df['e'].replace(0,-1), dtype='float64')
Y=y11.T
Y=Y.reshape(Y.shape[0],1)

one=-1*np.ones((Y.shape[0],1))
X1=np.array([df['b']], dtype='float64')
#X1=preprocessing.normalize(X1)
X1=X1.T


X2=np.array([df['a']], dtype='float64')
#X2=preprocessing.normalize(X2)
X2=X2.T

X=np.concatenate((X2,X1),axis=1)
                    
X=np.concatenate((X,one),axis=1)

a=0.004
i=30000
W=np.zeros(X[0].shape,dtype='float64')


W,errors=fit(X,Y,W,a,i)
#W=np.array([8.68216341, -7.35552523, 23.15316626])
print(W)
errors=np.array(errors)


plt.plot(errors[:,1],errors[:,0])

#red_patch = mpatches.Patch(color='blue', label='iris setosa')
#blue_patch = mpatches.Patch(color='green', label='iris versicolor and iris virginica')
#plt.legend(handles=[red_patch, blue_patch])


#plt.scatter(X[:,0],X[:,1],c=y,cmap = mcolors.ListedColormap(["blue", "green"]))

plt.title("The Iris Data Set", fontsize=18)
plt.xlabel(r'sepal length', fontsize=15)
plt.ylabel(r'sepal width', fontsize=15)


#plt.grid(True)

#xx=np.arange(4,8,0.2)
#yy=(+W[2]-xx*W[0])/W[1]
#plt.plot(xx,yy,c='r')


#yy1=(1+W[2]-xx*W[0])/W[1]
#yy2=(-1+W[2]-xx*W[0])/W[1]

#plt.plot(xx,yy1,c='b',linewidth=0.5)
#plt.plot(xx,yy2,c='b',linewidth=0.5)
plt.show()








