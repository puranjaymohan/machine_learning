import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, metrics

def loss(X,Y,W):
    pre=X.dot(W.T)
    if (Y.T.dot(pre)*1/Y.shape[0]) >=1.0 :
        return 0
    else:
        return 1-(Y.T.dot(pre)/Y.shape[0])




df=pd.read_csv('data.txt')

y=np.array(df['e'].replace(0,-1), dtype='float64')
Y=y.T
Y=Y.reshape(Y.shape[0])

one=-1*np.ones((Y.shape[0],1))
X1=np.array([df['b']], dtype='float64')
#X1=preprocessing.normalize(X1)
X1=X1.T


X2=np.array([df['a']], dtype='float64')
#X2=preprocessing.normalize(X2)
X2=X2.T

X=np.concatenate((X2,X1),axis=1)
#X=np.concatenate((X,one),axis=1)



def plot_classifier(svc):
    """
    function to plot the linear boundary made using the svm
    """
    w = svc.coef_[0]
    a = -w[0] / w[1]
    #xx = np.linspace(0.05, 0.12)

    xx = np.linspace(4, 8)
    yy = a * xx - (svc.intercept_[0]) / w[1] #equation of line of best fit(similiar to y=mx+c)
    plt.plot(xx, yy, 'k-')
    margin = 1 / np.sqrt(np.sum(svc.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin
    plt.plot(xx,yy_down, 'k-')
    plt.plot(xx,yy_up, 'k-')
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()

# we want to find the divider that classifies this data the best. lets try with c=1 first
#C=1000000
C=100
#here, we use the predefined svm function of sklearn and we use a linear kernel to fit the data
svc=svm.SVC(kernel='linear',C=C).fit(X,Y)# notice that one point is classified wrong
plot_classifier(svc)#plotting the classifier on the data


#xx=np.arange(0.05,0.12,0.01)
#yy=(+W[2]-xx*W[0])/W[1]
#plt.plot(xx,yy,c='b')


#yy1=(1+W[2]-xx*W[0])/W[1]
#yy2=(-1+W[2]-xx*W[0])/W[1]

#plt.plot(xx,yy1,c='g')
#plt.plot(xx,yy2,c='r')
#plt.show()








