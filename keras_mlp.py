#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 00:25:28 2018

@author: NikithaShravan
"""
import keras
import numpy as np
from keras.models import Sequential
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten


def loadData():
    
    num_classes = 10

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    
    num_rows = X_train.shape[1]
    num_cols = X_train.shape[2]
    num_channels = X_train.shape[3]
    input_dims = num_rows*num_cols*num_channels 
    
    
    X_train = X_train.reshape(X_train.shape[0], input_dims)
    X_test = X_test.reshape(X_test.shape[0], input_dims)
    
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
                          
    return X_train, Y_train, X_test, Y_test


def loadParams(activation='relu',learning_rate=1e-2, layer_dims=[12,7,5,6],
               dropout=False, dropout_val=0.15,optimizer='sgd',batch_size=20):
    return activation, learning_rate, layer_dims, dropout, dropout_val, optimizer, batch_size
    


def build_network(activation, layer_dims, input_dims, num_classes):
    """ 
    layer_dims: excluding input and output dimensions. 
    """
    model = Sequential()
    
    model.add(Dense(layer_dims[0],input_dim=input_dims))   
    #model.add(Dropout())
    
    for i in range(1,len(layer_dims)):
        model.add(Activation(activation))
        model.add(Dense(layer_dims[i]))
        #model.add(Dropout())
    
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
    return model

X_train, Y_train, X_test, Y_test = loadData()

assert(X_train.shape[0]==Y_train.shape[0])
assert(X_test.shape[0]==Y_test.shape[0])

input_dims = X_train.shape[1]
if Y_train.shape[1] > 1:
    num_classes = Y_train.shape[1]
    
else:
    num_classes = int(Y_train.max())

layer_dims  = [5]
model = build_network('relu', layer_dims, input_dims,num_classes)



model.fit(X_train, Y_train,
          batch_size=200,
          epochs=1000,
          verbose=1,
          validation_data=(X_test, Y_test))

from keras import backend as K
print("layers: ",model.layers[0].output)
for i in range(len(layer_dims)):
    layer_output = K.function([model.layers[0].input], [model.layers[i].output])
    print(layer_output([X_train])[0].shape)



score = model.evaluate(X_test, Y_test, verbose=0)
print(model.summary())
pred = model.predict_classes(X_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])



                      
                      



