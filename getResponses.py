#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 19:01:58 2018

@author: NikithaShravan
"""
""" Compute Features for any given image """ 
import numpy as np
from scipy import ndimage as ndi
#import matplotlib.pyplot as plt
#import matplotlib.image as img
from skimage import color


def rgb2gray(rgb):
    """ Convert RGB to GRAY SCALE """ 
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray    

def imageToVector(im):
    s = 1
    for i in im.shape:
        s *= i
    return np.reshape(im, (im.shape[0],im.shape[1]*im.shape[2]))
        
def getFilterResponses(im, filterSize=7, DogScales=[3,5], GaussianScales=[1]):
    """ im: Nx3 channel image , N: number of samples """ 
    print("Computing Lab images...")
    im = color.rgb2lab(im)
    responses = []
    num_channels = im.shape[3]
    for k in range(num_channels):
        for i in GaussianScales:
            a = ndi.gaussian_filter(im[:,:,:,k], sigma=i)
            responses.append(np.reshape(a, (a.shape[0], a.shape[1]*a.shape[2])))
#            print("responses size: ", np.shape(responses))
            
            b = ndi.laplace(a)
            responses.append(np.reshape(b, (b.shape[0], b.shape[1]*b.shape[2])))
        
        for i in DogScales:
            a = ndi.gaussian_gradient_magnitude(im[:,:,:,k], sigma=i)
            responses.append(np.reshape(a, (a.shape[0], a.shape[1]*a.shape[2])))

        for j in GaussianScales:
            
            t = ndi.gaussian_filter(im[:,:,:,k], sigma=i)
            a = ndi.sobel(t, axis=0)
            responses.append(np.reshape(a, (a.shape[0], a.shape[1]*a.shape[2])))
            b = ndi.sobel(t, axis=1)
            responses.append(np.reshape(b, (b.shape[0], b.shape[1]*b.shape[2])))
        
    return np.array(responses)

def getHOGFeatures(im):
    
    im = color.rgb2grey(im)
   
    from skimage.feature import hog
    hog_descriptor = hog(im, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(1, 1), feature_vector=True,block_norm='L2-Hys')
    return hog_descriptor
    
    