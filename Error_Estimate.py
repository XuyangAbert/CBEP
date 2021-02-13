# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:03:20 2018

@author: xyan
"""

import numpy as np

def Error_Estimate(Aggregated, Y_test):
    
    N = len(Y_test)
    
    Res = 0
    
    for i in range(N):
        
        Res += ((Y_test[i]-Aggregated[i])**2)
    
    V = 0
    
    MeanY = np.mean(Y_test)
    for j in range(N):
        
        V += (Y_test[i] - MeanY)**2
    
    Accuracy = 1 - Res / V
    
    return Accuracy