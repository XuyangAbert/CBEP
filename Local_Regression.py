# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:51:59 2018

@author: xyan
"""

from sklearn.neural_network import MLPRegressor
import numpy as np

def Local_Regression(X_Cluster,Y_Cluster, X_test):
    
    X = X_Cluster
    Y = np.array(Y_Cluster)
    
    
    
    clf =  MLPRegressor(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50,),
                        max_iter=200)
    
    clf = clf.fit(X, Y)
    
#    Y_pred = clf.predict(X_test)
    
    return clf
# Y_pred


