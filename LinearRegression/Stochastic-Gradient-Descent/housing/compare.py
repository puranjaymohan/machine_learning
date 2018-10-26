import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


thetaf=np.array([[162.01523275,7778.81131956]], dtype=float)
thetaf2=np.array([[1.65360934e+02,9.39876086e-02]], dtype=float)

plt.xlabel('Area of House')
plt.ylabel('Price of House')
xx=np.arange(500,4500)
yy=xx*thetaf[0,0]+thetaf[0,1]
plt.plot(xx,yy)




xxx=np.arange(500,4500)
yyy=xxx*thetaf2[0,0]+thetaf2[0,1]
plt.plot(xxx,yyy)




plt.show()

