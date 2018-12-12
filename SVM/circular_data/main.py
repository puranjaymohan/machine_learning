import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, metrics

data=pd.read_csv('data.csv')


y=np.array(data['target'].replace(2.0,-1))


x=np.array(data[['x1','x2']])
C=1
clf=svm.SVC(kernel='rbf',C=C,gamma=0.7)

clf.fit(x,y)


x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
plt.scatter(x[:, 0], x[:, 1], c=y) #cmap=plt.cm.Paired)
plt.title('Toy Dataset with RBF Kernel')
plt.show()
