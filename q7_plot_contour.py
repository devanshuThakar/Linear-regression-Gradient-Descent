import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from linearRegression.linearRegression import LinearRegression
from sklearn.metrics import mean_squared_error 

N=100
x = pd.Series(np.linspace(-1., 1., N))
y = pd.Series( 3*x + 2*np.random.randn(N))

writer = FFMpegWriter(fps=2)
fig=plt.figure()
ax = fig.add_subplot()
plt.scatter(x,y)
plt.xlabel("x")
plt.ylabel("y")
l, = plt.plot([],[])

with writer.saving(fig, "Images/Q7_gif.gif",100):
    for iteration in range(1,11):
        reg = LinearRegression()
        reg.fit_non_vectorised(x,y,20,iteration,lr=0.001,lr_type='constant')
        y_hat = reg.predict(x)
        # print(iteration, mean_squared_error(y,y_hat))
        l.set_data(x,y_hat)
        ax.set_title("Iteration-{} MSE = {:.2f}".format(iteration, mean_squared_error(y_hat,y)))
        writer.grab_frame()