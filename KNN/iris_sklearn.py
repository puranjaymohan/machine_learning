import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)


iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=32)





predicts=[]
for i in range(1,20):
    
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    predicts.append(accuracy_score(y_test,neigh.predict(X_test)))

xpre=np.arange(1,20)

neigh = KNeighborsClassifier(n_neighbors=8)
neigh.fit(X, y)

####
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5


h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


xnew=np.concatenate((xx.ravel().reshape((28200,1)),yy.ravel().reshape((28200,1))),axis=1)
print(xnew.shape)

ynew=np.asarray(neigh.predict(xnew))
print(ynew.shape)

plt.figure(2, figsize=(12, 6))

plt.clf()

plt.subplot(121,aspect='equal')

plt.scatter(xnew[:,0],xnew[:,1],c=ynew,cmap='brg')








plt.scatter(X[:, 0], X[:, 1], c=y,s=40,cmap='bone')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.subplot(122)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.plot(xpre,np.asarray(predicts))
plt.xticks(xpre)
plt.yticks(predicts)
plt.show()




