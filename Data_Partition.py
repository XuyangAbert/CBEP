# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:58:14 2018

@author: xyan
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
def Data_Partition(TrainData):
    # Compute the size of the Training Data
    [rows,cols]=np.shape(TrainData)
    # Extract the input variables columns
    Train_X = TrainData[0:rows]
    
    # Initialize the number of clusters as two
    k = 2
    
    while 1:
        # Implement the K-means clustering procedure on Input variables
        
        kmeans = KMeans(n_clusters=k, random_state=0).fit(Train_X)
        # Extract the labels and centers of clusters
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        # Initialize the Inner-Cluster distance vector
        InnerDist = np.zeros(k)
        # Claculate the sum of the squred distance for each cluster
        for i in range(k):
            
            Cluster_Indice = i
            
            SumD = 0
            count = 0
            
            for j in range(len(labels)):
                
                if labels[j] == Cluster_Indice:
                    
                    SumD += distance.euclidean(Train_X[j][0:], centers[i][0:])
                    count += 1
            
            InnerDist[i] = SumD/count
            
        InterDist = distance.pdist(centers)
        
        # Compute the total summation of the Inner-Cluster distance
        TotalInnerDist = np.sum(InnerDist)
        # Compute the total summation of the Inter-Cluster distance
        TotalInterDist = np.sum(InterDist)
        
        # Cluster Separability
        Sep = TotalInnerDist / (TotalInnerDist + TotalInterDist)
        # Check whether the TotalInner-Cluster distance increases or not 
        if k > 8:
            # If the TotalInner-Cluster distance starts to decrease, then exit the loop
            if Sep <= Sep_old:
                break
            
                
        else:
        # Update the Temporay holder for the previous TotalInner-Cluster distance
            Sep_old = Sep
        # Increase number of clusters by one 
            k += 1
        
        
    
    # Return the optimal clustering result:cluster lables and centers
    return centers, labels
    
    
    
