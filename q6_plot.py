from cProfile import label
import encodings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

x=x.reshape((x.shape[0],1))
y=y.reshape((y.shape[0],1))
Data = np.hstack((x,y))
degrees = [1,3,5,7,9]
N_list = [10,20,30,40,50,60]

for N in N_list:
    Magnitude_theta=[]
    for d in degrees:
        poly = PolynomialFeatures(d)
        idx = np.random.randint(Data.shape[0],size=N)
        x_=Data[idx,0].reshape(Data[idx,0].shape[0],1)
        x_=poly.transform(x_)
        y_=Data[idx,1]
        reg=LinearRegression()
        reg.fit(x_,y_)
        theta=reg.coef_
        Magnitude_theta.append(np.sqrt(np.dot(theta,theta)))
    plt.semilogy(degrees,Magnitude_theta, marker='o', label='N={}'.format(N))

plt.legend()
plt.title(r'Plot of $\sqrt{\theta^T \theta}$ vs. degre $d$')
plt.ylabel(r'$\sqrt{\theta^T \theta}$')
plt.xlabel(r'degree $d$')
plt.savefig('Images/q6_plot_N{}.png'.format(N_list[-1]))
plt.show()