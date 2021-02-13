# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:57:35 2018

@author: xyan
"""

if __name__ == "__main__":
    from Local_Regression import Local_Regression
    from Data_Partition import Data_Partition
    from Single_Regression import Single_Regression
    from ClusterOfLMS import ClusterOfLMS
    from Error_Estimate import Error_Estimate
    from sklearn.model_selection import train_test_split
#    import matplotlib.pyplot as plt
#    from matplotlib.colors import ListedColormap
    import pandas as pd
    import numpy as np
    
    df= pd.read_csv('Electric.csv', header = None)
    
    X = df.iloc[0:, 0:8].values
    Y = df.iloc[0:, 8].values
    
    
       
    
    Y = np.array(Y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, 
                                                        random_state=1)
    
    # Single Regression Tree using the training set without partitioning
    BaseModel = Single_Regression(X_train, y_train)
    
    # Accuracy Evaluation of the Single Global Regression Model
    BaseAccuracy = BaseModel.score(X_test, y_test)
    
    [centers, labels] = Data_Partition(X_train)
    
    # The clustered regression trees   
    GModel = []
    
        
    sample = X_test
        
    for i in range(np.shape(centers)[0]):
            
        X_Cluster = X_train[labels == i]
        Y_Cluster = y_train[labels == i]
            
        # Define local model as a class
        LM = Local_Regression(X_Cluster,Y_Cluster, X_test)
        
        GModel.append(LM)
        
    Aggregated = ClusterOfLMS(GModel, X_test, y_test, centers, X_train, y_train)
    
    GModel_Accuracy = Error_Estimate(Aggregated, y_test)
    
#    plt.bar(x=['BaseAccuracy-ANN', 'GModel_Accuracy-ANN'], height = [BaseAccuracy, 
#            GModel_Accuracy], width = 0.3, color = ['r','b'] )
#    plt.yticks(np.arange(0.0,1.0,0.1))
#    
#    plt.show()
    
    