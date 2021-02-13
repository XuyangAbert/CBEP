# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:59:38 2018

@author: xyan
"""


from sklearn.neural_network import MLPRegressor

def Single_Regression(X_Train, Y_Train):
    
    X = X_Train
    Y = Y_Train
    
    clf = MLPRegressor(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50,),
                        max_iter=200)
    
    # Fit the training data set into the MLP Classifier
    clf = clf.fit(X, Y)
#    clf = tree.DecisionTreeClassifier()
#    
#    clf = clf.fit(X,Y)
    
    return clf