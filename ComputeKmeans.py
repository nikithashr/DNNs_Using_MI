#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:04:12 2018

@author: NikithaShravan
"""

""" Compute K-Means to get Visual Words """
from data_utils import load_CIFAR10
from getResponses import getFilterResponses, getHOGFeatures
from sklearn.cluster import MiniBatchKMeans 
import numpy as np
import time
from sklearn.metrics.pairwise import pairwise_distances_argmin

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

XData, yData = load_CIFAR10('datasets/cifar-10-batches-py')

def ComputeKMeans(responses, K=50):
    mbk = MiniBatchKMeans(K,batch_size=100)
    mbk.fit(responses)

    return mbk.cluster_centers_ 

def computeImageFeatures(XData, K=50):
    
    """ choosing fewer training samples to run the code faster.
    Run the entire dataset on Ada """
    num_samples = XData.shape[0]
    
    print("Input data dimensions: ", XData[:num_samples ].shape)
    responseAllImages = []

    responseAllImages = getFilterResponses(XData[:num_samples])
    print("Filter Responses dimension: ", responseAllImages.shape)
        
    input_dim = responseAllImages.shape[0]
    N = responseAllImages.shape[1]
    pixels_per_image = responseAllImages.shape[2]
    reshapedResponses = np.reshape(responseAllImages,(input_dim, N*pixels_per_image)).T
    clusters = ComputeKMeans(reshapedResponses,K)
    print("Cluster center dimension: ", clusters.shape)
    labels = pairwise_distances_argmin(reshapedResponses, clusters)
    labels = np.reshape(labels, (N,pixels_per_image))
    
    """ parallelizing the code to make histogram computation more efficient
    ref: https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis/44155607#44155607 
    """
    bins = np.linspace(1,K,K)
    idx = np.searchsorted(bins, labels,'right')
    
    scaled_idx = K*np.arange(labels.shape[0])[:,None] + idx
    limit =K*labels.shape[0]
    counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
    hist = np.reshape(counts, (labels.shape[0],K))
    
    print("Histogram dimension: ", hist.shape)

    return hist

def computeHoGFeatures(XData):
    results = Parallel(n_jobs=num_cores)(delayed(getHOGFeatures)(XData[i]) for i in range(XData.shape[0]))
    hog_descriptor = np.array(results)
    print("hog: ", hog_descriptor.shape)
#    return results
    
      #Nxpx36
    hog_descriptor = np.reshape(hog_descriptor,(XData.shape[0],16,9))
    input_dim = hog_descriptor.shape[0]
    N = hog_descriptor.shape[1]
    pixels_per_image = hog_descriptor.shape[2]
    hog_ordered = np.transpose(hog_descriptor, (2,0,1))
    reshapedResponses = np.reshape(hog_ordered,(pixels_per_image,input_dim*N)).T
    K = 50
    clusters = ComputeKMeans(reshapedResponses,K)
    print("Cluster center dimension: ", clusters.shape)
    labels = pairwise_distances_argmin(reshapedResponses, clusters)
    labels = np.reshape(labels, (input_dim, N))
    
    bins = np.linspace(1,K,K)
    idx = np.searchsorted(bins, labels,'right')
    
    scaled_idx = K*np.arange(labels.shape[0])[:,None] + idx
    limit =K*labels.shape[0]
    counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
    hist_hog = np.reshape(counts, (labels.shape[0],K))

    return hist_hog

hist_hog = computeHoGFeatures(XData)

import scipy.io as sio
sio.savemat('histogram_hog.mat',{'hist_hog':hist_hog})


hist_texture = computeImageFeatures(XData)

sio.savemat('histogram_texture.mat', {'hist_texture':hist_texture})