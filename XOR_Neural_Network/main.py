import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X= np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
Y=np.array([[0],
            [1],
            [1],
            [0]])


def activate(Z):
    Z= np.concatenate(((1/(1+np.exp(-Z))),np.ones((Z.shape[0],1))),axis=1)
    return Z

def sigmoidgrad(A):
    s=1/(1+np.exp(-A))
    return s*(1-s)

def sigmoid(A):
    return 1/(1+np.exp(A))

theta1 = (np.random.random(( 3, 2)))
theta2 = (np.random.random((3, 1)))

def forward(X,theta1,theta2):
    a1=X
    z1=a1.dot(theta1)
    a2=activate(z1)
    z2=a2.dot(theta2)
    hyp=sigmoid(z2)
    return a1,z1,a2,z2,hyp


###Back Propagation###

lr=1;
mse=[]
for i in range(50000):
    for s in range(X.shape[0]):    
        a1, z1, a2, z2, hyp = forward(X[s,:].reshape(1,3), theta1, theta2)
        error=Y[s]-hyp
        delta2=(error*sigmoidgrad(z2)*a2).T
        delta1=(error*sigmoidgrad(z2)*np.multiply(theta2[0:2,:],sigmoidgrad(z1).T).dot(a1)).T
        #Gradient Checking code
        #a1, z1, a2, z2, hyp1 = forward(X[s,:].reshape(1,3), theta1, theta2+0.001)
        #a1, z1, a2, z2, hyp2 = forward(X[s,:].reshape(1,3), theta1, theta2-0.001)
        #print(delta1.shape)
        #print(np.sum(delta2),'--')
        #print( (((1/2)*(Y[s]-hyp1)**2)-((1/2)*(Y[s]-hyp2)**2))/0.002 )

        theta1 -= lr*delta1
        theta2 -= lr*delta2
    
    
    a1, z1, a2, z2, hyp = forward(X, theta1, theta2)
    if i%1000 ==0:    
        print((1/8)*np.sum((Y-hyp)**2))
    mse.append([i,(1/8)*np.sum((Y-hyp)**2)])

a,b,c,d,e = forward(X,theta1,theta2)
print(e)
mse=np.array(mse)
plt.figure(figsize=[10,5])
plt.subplot(121)
plt.plot(mse[:,0],mse[:,1])
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.subplot(122)


h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


xnew=np.concatenate((xx.ravel().reshape(4900,1),yy.ravel().reshape(4900,1)),axis=1)

xnew1=np.concatenate((xnew,np.ones((4900,1))),axis=1)
a,b,c,d,e = forward(xnew1,theta1,theta2)
plt.scatter(xnew[:,0],xnew[:,1],c=e.reshape(4900))

plt.scatter(X[0,0],X[0,1],c='w',edgecolors='w',marker='x',s=60)
plt.scatter(X[1,0],X[1,1],c='w',edgecolors='w',marker='o',s=60)
plt.scatter(X[2,0],X[2,1],c='w',edgecolors='w',marker='o',s=60)
plt.scatter(X[3,0],X[3,1],c='w',edgecolors='w',marker='x',s=60)
plt.show()

