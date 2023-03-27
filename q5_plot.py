import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

x=x.reshape((x.shape[0],1))
Magnitude_theta=[]
degrees = [i for i in range(1,10)]
for d in degrees:
    poly = PolynomialFeatures(d)
    x_=poly.transform(x)
    reg=LinearRegression()
    reg.fit(x_,y)
    theta=reg.coef_
    Magnitude_theta.append(np.sqrt(np.dot(theta,theta)))

plt.semilogy(degrees,Magnitude_theta, color='blue', marker='o', linestyle='solid',
     linewidth=2, markersize=8)
plt.title(r'Plot of $\sqrt{\theta^T \theta}$ vs. degre $d$')
plt.ylabel(r'$\sqrt{\theta^T \theta}$')
plt.xlabel(r'degree $d$')
plt.savefig('Images/q5_plot_magnitude_theta_vs_degree.png')
plt.show()