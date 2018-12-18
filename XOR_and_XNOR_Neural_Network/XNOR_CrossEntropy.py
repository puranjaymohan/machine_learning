import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X= np.array([[0,0,1,1],
            [0,1,0,1]])
Y=np.array([[1,0,0,1]])



b1=np.random.random()
b2=np.random.random()

def sigmoidgrad(A):
    s=1/(1+np.exp(-A))
    return s*(1-s)

def sigmoid(A):
    return 1/(1+np.exp(-A))

theta1 = (np.random.random(( 2, 2)))
theta2 = (np.random.random((2, 1)))

def forward(X,theta1,theta2,b1,b2):
    a1=X
    z1=theta1.T.dot(a1)+b1
    a2=sigmoid(z1)
    z2=theta2.T.dot(a2)+b2
    hyp=sigmoid(z2)
    return a1,z1,a2,z2,hyp


###Back Propagation###

lr=3;
mse=[]
for i in range(1900):
   
    a1, z1, a2, z2, hyp = forward(X, theta1, theta2,b1,b2)
    dw2=a2.dot((hyp-Y).T)
    dw1=np.multiply(theta2.dot(hyp-Y),sigmoidgrad(z1)).dot(a1.T)
    db2=hyp-Y
    db1=np.multiply(theta2.dot(hyp-Y),sigmoidgrad(z1))
    #Gradient Checking code
    #a1, z1, a2, z2, hyp1 = forward(X, theta1, theta2,b1+0.001,b2)
    #a1, z1, a2, z2, hyp2 = forward(X, theta1, theta2,b1-0.001,b2)
    
    #print(np.sum(db1),'--')
    #print( np.sum((-(Y*(np.log(hyp1))+(1-Y)*(np.log(1-hyp1))))-(-(Y*(np.log(hyp2))+(1-Y)*(np.log(1-hyp2)))),axis=1)/0.002 )

    theta1 -= lr*dw1
    theta2 -= lr*dw2
    b1 -= lr*np.sum(db1)
    b2 -= lr*np.sum(db2)
    
    
    a1, z1, a2, z2, hyp = forward(X, theta1, theta2,b1,b2)
    if i%100 ==0:    
        print((1/8)*np.sum((Y-hyp)**2))
    mse.append([i,(1/8)*np.sum((Y-hyp)**2)])

a,b,c,d,e = forward(X,theta1,theta2,b1,b2)
print(e)

mse=np.array(mse)
plt.figure(figsize=[10,5])
plt.subplot(121)
plt.plot(mse[:,0],mse[:,1])
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.subplot(122)


h = .02  # step size in the mesh
x_min, x_max = X[0, :].min() - 0.2, X[0, :].max() + 0.2
y_min, y_max = X[1, :].min() - 0.2, X[1, :].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


xnew=np.concatenate((xx.ravel().reshape(1,4900),yy.ravel().reshape(1,4900)),axis=0)


a,b,c,d,e = forward(xnew,theta1,theta2,b1,b2)
plt.scatter(xnew[0,:],xnew[1,:],c=e[0,:])

plt.scatter(X[0,0],X[1,0],c='w',edgecolors='w',marker='o',s=60)
plt.scatter(X[0,1],X[1,1],c='w',edgecolors='w',marker='x',s=60)
plt.scatter(X[0,2],X[1,2],c='w',edgecolors='w',marker='x',s=60)
plt.scatter(X[0,3],X[1,3],c='w',edgecolors='w',marker='o',s=60)
plt.show()

