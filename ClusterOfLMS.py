# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:37:39 2018

@author: xyan
"""


from scipy.spatial import distance
from numpy.linalg import inv
import numpy as np

def ClusterOfLMS(LMS, X_test, Y_test, centers, X_Train, Y_Train):
    # Initialize the local prediction matrix
    LP = np.zeros((len(LMS), len(Y_test)))
    # Initialze the testing error as zero
#    Error = 0
    # Initialize the aggregated prediction vector
    Y_Pred = []
    
    W1 = []
    
    # Fill the empty local prediction matrix 
    for i in range(len(LMS)):
        
        LM = LMS[i]
        
        LP[i][0:] = LM.predict(X_test)
        
        W1.append(LM.score(X_Train, Y_Train))
        
        
    # Obtain the aggregated prediction vector from the local prediction matrix
        
    for j in range(len(Y_test)):
        dist = []
        for k in range(len(LMS)):
            
            dist.append(distance.euclidean(X_test[j], centers[k]))
        
        
        # Compute the weight for each local regression model    
        W2 = dist / np.sum(dist)
        W2 = np.exp(-W2)
#        Weight = dist / np.sum(dist)
        # W1 = np.abs(W1)/np.sum(np.abs(W1))
        # Obtain the aggregated predicted output for the jth test instance
        # Weight = np.multiply(W1, W2)
        Weight = W2 / np.sum(W2)
        inde = np.argmax(Weight)
        Temp = 0
        for l in range(len(LP)):
            Temp += (LP[l,j] * Weight[l])
        Y_Pred.append(LP[inde,j])
    return Y_Pred
        
        
        
    
        
        
    