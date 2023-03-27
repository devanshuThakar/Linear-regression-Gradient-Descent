from email import message
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad
from sklearn.metrics import mean_squared_error 
# Import Autograd modules here

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

        pass

    def fit_non_vectorised(self, X, y, batch_size=1, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        #Using MSE for gradient

        count=0
        x=X.to_numpy()
        if self.fit_intercept==True:
            x=np.c_[np.ones(x.shape[0]),x]#Adding ones
        x=x.tolist()
        #Turning into lists for better accessability
        
        y=y.to_numpy().tolist()
        n_samples=len(y)
        n_features=len(x[0])#m+1  including bias term
        mini_batchx=[]
        mini_batchy=[]

        # thetas=np.ones(n_features)#initialise
        thetas = np.random.randn(n_features)

        for i in range(n_iter):
            
            if lr_type=='constant':
                alpha=lr
            else:
                alpha=lr/(i+1)

            if (count+1)*batch_size>=n_samples:
                mini_batchx=x[count*batch_size:n_samples]
                mini_batchy=y[count*batch_size:n_samples]
                count=-1
            else:
                mini_batchx=x[count*batch_size:(count+1)*batch_size]
                mini_batchy=y[count*batch_size:(count+1)*batch_size]
            
            
            count=count+1

            grad=np.zeros(n_features)
            
            for j in range(len(mini_batchy)):
                error=mini_batchy[j]
                for k in range(n_features):
                    error=error-thetas[k]*mini_batchx[j][k]
                
                for k in range(n_features):
                    grad[k]=grad[k]+(2/n_samples)*(error *mini_batchx[j][k])

            for k in range(n_features):
                thetas[k]=thetas[k]- alpha*grad[k]
        
        self.coef_=thetas

    def fit_vectorised(self, X, y,batch_size=1, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        #Using MSE for gradient

        count=0
        x=X.to_numpy()
        if self.fit_intercept==True:
            x=np.c_[np.ones(x.shape[0]),x]#Adding ones
        x=x.tolist()
        #Turning into lists for better accessability
        
        y=y.to_numpy().tolist()
        n_samples=len(y)
        n_features=len(x[0])#m+1  including bias term
        mini_batchx=[]
        mini_batchy=[]

        thetas=np.ones(n_features)#initialise

        for i in range(n_iter):
            
            if lr_type=='constant':
                alpha=lr
            else:
                alpha=lr/(i+1)

            if (count+1)*batch_size>=n_samples:
                mini_batchx=x[count*batch_size:n_samples]
                mini_batchy=y[count*batch_size:n_samples]
                count=-1
            else:
                mini_batchx=x[count*batch_size:(count+1)*batch_size]
                mini_batchy=y[count*batch_size:(count+1)*batch_size]
            
            
            count=count+1
            mini_batchx=np.array(mini_batchx)
            mini_batchy=np.array(mini_batchy)

            thetas=thetas - (2*alpha/n_samples)*(mini_batchx.transpose()@mini_batchy-(mini_batchx.transpose()@mini_batchx@thetas))
        
        self.coef_=thetas

    def mse(self,theta,x,y,n_samples):
        e=y-x@theta
        mse=(1/n_samples)*(e.transpose()@e)
        return mse

    def fit_autograd(self, X, y, batch_size=1, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        #Using MSE for gradient

        count=0
        x=X.to_numpy()
        if self.fit_intercept==True:
            x=np.c_[np.ones(x.shape[0]),x]#Adding ones
        x=x.tolist()
        #Turning into lists for better accessability
        
        y=y.to_numpy().tolist()
        n_samples=len(y)
        n_features=len(x[0])#m+1  including bias term
        mini_batchx=[]
        mini_batchy=[]

        thetas=np.random.randn(n_features)#initialise

        for i in range(n_iter):
            
            if lr_type=='constant':
                alpha=lr
            else:
                alpha=lr/(i+1)

        
            if (count+1)*batch_size>=n_samples:
                mini_batchx=x[count*batch_size:n_samples]
                mini_batchy=y[count*batch_size:n_samples]
                count=-1
            else:
                mini_batchx=x[count*batch_size:(count+1)*batch_size]
                mini_batchy=y[count*batch_size:(count+1)*batch_size]
            
            count=count+1
            mini_batchx=np.array(mini_batchx)
            mini_batchy=np.array(mini_batchy)
            grad=egrad(self.mse)(thetas,mini_batchx,mini_batchy,n_samples)
           
            thetas=thetas - alpha*grad

        self.coef_=thetas
        pass

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        x=X.to_numpy()
        y=y.to_numpy()

        self.coef_=np.linalg.inv(x.transpose()@x)@(x.transpose()@y)


    def predict(self, X,isvectorized=True):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        x=X.to_numpy()
        if self.fit_intercept==True:
            x=np.c_[np.ones(x.shape[0]),x]#Adding ones

        thetas=self.coef_
        if isvectorized==False:
            y_hats=[]
            for i in range(x.shape[0]):
                y_hat=0
                for j in range(x.shape[1]):
                    y_hat=y_hat+x[i,j]*thetas[j]
                y_hats.append(y_hat)
        else:
            y_hats=x@thetas
            y_hats=y_hats.tolist()

        y_predict=pd.Series(y_hats)
        return y_predict

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """

        pass

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """

        pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        pass
