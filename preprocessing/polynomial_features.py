''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
# from itertools import chain, combinations
# from itertools import combinations_with_replacement as combinations_w_r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialFeatures():
    
    def __init__(self, degree=2,include_bias=True):
        """
        Inputs:
        param degree : (int) max degree of polynomial features
        param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        self.degree=degree
        self.include_bias=include_bias    
        pass

    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """
        # comb = combinations
        # n_features = X.shape[1]
        # iter = chain.from_iterable(combinations(range(n_features),i) for i in range(1,self.degree+1))
        # if(self.include_bias):
        #     iter = chain(comb(range(n_features),0), iter)
        # iter=list(iter)
        # X_ = np.ones((X.shape[0],len(iter)))

        # for ind,combination in enumerate(iter):
        #     if(len(combination)==0):
        #         continue
        #     temp=np.ones((X.shape[0]))
        #     for i in combination:
        #         temp = temp*X[:,i]
        #     X_[:,ind]=temp
        
        n_features = X.shape[1]
        n_new_features = self.degree*n_features
        i=0
        if(self.include_bias):
            n_new_features+=1
            i=1
        X_=np.ones((X.shape[0],n_new_features))

        
        for degrees in range(1,self.degree+1):
            for m in range(n_features):
                temp=np.ones((X.shape[0]))
                for d in range(1,degrees+1):
                    temp=temp*X[:,m] 
                X_[:,i]=temp
                i+=1


        return X_          