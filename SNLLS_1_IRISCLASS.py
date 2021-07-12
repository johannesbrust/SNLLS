#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SNLLS_1_IRISCLASS.py
#------------------------------------------------------------------------------
# Script to test 6 algorithms on a least-squares classification problem.
# The full problem is 
#
#        minimize f(w)   = ||r(w)||^2 / 2K
#                        = ( sum ri^2 ) / 2K
#                        = ( sum (model(x_i)-y_i)^2 ) / 2K,                 (1)
#                            
#        where f: R^n->R (nonlinear least-squares NLLS) objective/loss, 
#        r: R^n->R^N and "K" is a constant for averaging. Typically not all of
#        the data (x_i,y_i) i \in [0,N] is available at once, because of large 
#        sizes. Instead smaller "batched" portions of the data are used.
#        The data is divided into N=L*M with separation
#        
#                f(w)    = ( sum fj(w) ),     j \in [1,M]
#                
#        where
#
#                fj(w)   = ( sum (model(x_l)-y_l)^2 ) / 2K,  
#                
#        and l \in [(j-1)*L,j*L]. One evalution of a loss function 
#        (using e.g., TensorFlow) is
#        
#            2K/L * fj(w)  = "loss_value" = ( sum (model(x_l)-y_l)^2 ) / L  (2)
#
#                          
#        This script compares "stochastic" algorithms using the information in
#        (2) to solve the problem in (1). The algorithms are
#        
#        NLLS (Full Jacobian nonlinear least-squares)
#        SNLLS1 (rank-1 stochastic jacobian least-squares)
#        SNLLSL (rank-L stochastic jacobian least-squares)
#        SGD 
#        ADAM 
#        ADAGRAD
#------------------------------------------------------------------------------
#    MODIFICATIONS: 
#        01/18/21, X.X., Including "MeanSquaredError()"
#        01/19/21, X.X., Using "softmax", and "one_hot" from TF
#        02/03/21, X.X., Using the TF jacobian
#        03/01/21, X.X., Using a non-linear least squares update
#        03/09/21, X.X., Comparing outcomes from SGD and NLLS solver
#        03/12/21, X.X., Trying Stochastic Jacobian
#        03/15/21, X.X., Switching-off experimental "parfor" for speed-up 
#        06/21/21, X.X., Writing results to files
#        07/01/21, X.X., Modifications for consistency 
"""

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Setup program

### General imports
import time
import os
import tensorflow as tf
import numpy as np

### Solver imports
import NLLS
import SNLLS1
import SNLLSL

# Setup for storing data
dataPath = './DATA/'
filePrefix = 'EX_1'
numRuns = 5 # 10

# Setting logging level
#tf.get_logger().setLevel('ERROR')

## DATA
# [dataset of 120 Iris flowers](https://en.wikipedia.org/wiki/Iris_flower_data_set) 
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

features, labels = next(iter(train_dataset))

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

features, labels_ = next(iter(train_dataset))

# Define labels as "one_hot" indicators to use MSE loss
# 01/19/21, Modification, X.X.
labels = tf.one_hot(labels_,3)

## MODEL DEFINITION

# Modification of output layer to include a "softmax" probability
# 01/19/21, Modification, X.X.

# Number of algorithms in this experiment
# Algs. 1--3 correspond to NLLS methods, and 4--6 correspond to known methods
numAlgs = 6

# Initialize one model per algorithm
models = []
for i in range(numAlgs):
    models.append(tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(3, activation=tf.nn.softmax)]))


# Initialize "model0" by calling on it
predictions = models[0](features)

# Using a mean squared loss function, 01/18/21
loss_object = tf.keras.losses.MeanSquaredError()

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  return loss_object(y_true=y, y_pred=y_)

# Assign same initial variables to all models and
# print losses 
lossesL = [] #np.zeros((numAlgs,1))
for m in range(numAlgs):
    for i in range(len(models[m].trainable_variables)):
        models[m].trainable_variables[i].assign(models[0].trainable_variables[i])
    lossesL.append(loss(models[m],features,labels,training=True))
        #losses[m] = loss(models[m],features,labels,training=True)
    
#l1 = loss(model1, features, labels, training=True)
print("\nLoss test (Initial): \n")
 
solverFormat = "NLLS    SNLLS1  SNLLSL  SGD     ADAM    ADAGRAD \n"
lossFormat = ("{0[0]:.4f}  {0[1]:.4f}  {0[2]:.4f}  {0[3]:.4f}"
                "{0[4]:.4f}  {0[5]:.4f} \n")
                                
#print("NLLS (New Alg.) \t SGD (Old Alg.) \n")
print("Epoch \t"+solverFormat)
print("Init. \t"+lossFormat.format(lossesL))
#print("{} \t {} \n".format(l,l1))

# Gradient to define the training algorithm
# The nonlinear least-squares methods will have additional
# gradient computation functions, because Jacobians (or --
#     stochastic approximations) are also included
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

# This part defines the gradient and Jacobian computation 
# (with initial evaluation at iteration k = 0)
# To be used by model0
k = 0

# Functions "gradJac" and "update_step"
# This function call makes sure that all is in order
batch_size_Jac = 96 # 32 (Value 96 corresponds to the full data)
loss_value, grads, errsV, jacV, idxJac = NLLS.gradJac(models[0],features,labels,batch_size_Jac)


### PROBLEM DIMENSION
# Counter for number of derivative computations, gradients and Jacobians
numSolv = 0  

numData = len(idxJac)
numLays = len(grads)
varDims = np.zeros((numLays,2))
numVars = 0
shps = []
for i in range(numLays):
    shpi = grads[i].shape
    shps.append(shpi)
    numVars = numVars + np.prod(shpi)

# Print size of variables and Data (for batch)
print("\nSize variables: {}, Size data: {}".format(numVars,
                                          numData))

## KNOWN OPTIMIZER
learning_rate = 1.0        
# SGD
optimizerSG = tf.keras.optimizers.SGD(learning_rate=learning_rate)#(learning_rate=0.01)

# ADAM
optimizerAD = tf.keras.optimizers.Adam()#(learning_rate=0.01)

# ADAGRAD
optimizerAG = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)#(learning_rate=0.01)
    

# PARAMTERS

# SNLLS1 (model2)

# Indices for stoch. Jacobian
numData0 = numData
idxE = tf.constant(range(numData0))
idxJ = tf.constant(range(numVars))
nDm1 = numData0-1

# Jacobian accumulation
jk = tf.Variable(np.zeros(numVars),dtype=tf.float32)
lk = 0
delta = 0.8 # np.sqrt(numData/(2*(numSamples/batchSize)))
delta1 = 1.0

rho = 0.0 # 0.0005 # 0.0005 Parameter for error terms added to the diagonal
ek = np.zeros(numVars)
gDiag2 = tf.ones(numVars)
gDiag2 = 0.0000000001 + tf.zeros(numVars)
#beta2 = 1 #7
beta2 = 0.05

# SNLLSL (model3)

#beta3 = 1#learning_rate # 0.5 0.8 #learning_rate #1#learning_rate

beta3 = 0.05
gDiag3 = 0.0000000001 + tf.zeros(numVars)
jacs3 = tf.Variable(np.zeros(numVars),dtype=tf.float32)

# Storing intermediate data values
train_loss_results = []
train_accuracy_results = []
train_loss_results1 = []
train_accuracy_results1 = []

# Timings
times_ave_Jacs = []
times_ave_Updates = []

num_epochs = 50

for run in range(numRuns):
    
    print("\nRun:"+str(run)+"\n")
    print("Epoch \t"+solverFormat)
    
    # Store loss values
    fileName = filePrefix+'_LOSS_RUN_'+str(run)
    fileW = open((dataPath+fileName+'.txt'),'w')
    
    # Reinitialize variables
    if run > 0:
        models.clear()
        for i in range(numAlgs):
            models.append(tf.keras.Sequential([
                    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
                    tf.keras.layers.Dense(10, activation=tf.nn.relu),
                    tf.keras.layers.Dense(3, activation=tf.nn.softmax)]))
        
        
        idxE = tf.constant(range(numData0))
        idxJ = tf.constant(range(numVars))
        nDm1 = numData0-1
        
        # Variables being accumulated
        jk = tf.Variable(np.zeros(numVars),dtype=tf.float32)
        lk = 0
        ek = np.zeros(numVars)        
        gDiag2 = 0.0000000001 + tf.zeros(numVars) #tf.ones(numVars)
        gDiag3 = 0.0000000001 + tf.zeros(numVars) #tf.ones(numVars)
        jacs3 = tf.Variable(np.zeros(numVars),dtype=tf.float32)

    for epoch in range(num_epochs):
      epoch_loss_avg = tf.keras.metrics.Mean()
      # Modification of loss measure
     # epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
      epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
      
      # For model1
      epoch_loss_avg1 = tf.keras.metrics.Mean()
      epoch_accuracy1 = tf.keras.metrics.SparseCategoricalAccuracy()  
      
      # Loss values for storings
      losses = np.zeros((numAlgs))
      
      # Training loop - using batches of 32
      # Timing data
      tJs = 0.0
      tUs = 0.0
      denum = 0
      for x, y in train_dataset:
        # Convert labels  
        y_one = tf.one_hot(y,3)  
        numData = np.prod(y_one.shape)
    
        #---------------------- Optimize the model---------------------------------
        # This uses the "nonlinear least squares (NLLS)" update_step function
        # and gradient and Jacobian computations
            
        # Compute derivatives
        #loss_value, grads, errsV, jacV = gradJac(model,features,labels)
        tsJ = time.time()
        #loss_value, grads, errsV, jacV = gradJac(model,x,y_one)
        loss_value, grads, errsV, jacV, idxJac = NLLS.gradJac(models[0],x,y_one,numData) # batch_size_Jac
        k = k + 1
        teJ = time.time()
        tJ = teJ-tsJ
        
        # Updated step
        tsU = time.time()
        # numData
        NLLS.update_step(models[0],grads,jacV,numData,numVars,shps) # batch_size_Jac
        numSolv = numSolv + 1
        teU = time.time()
        tU = teU-tsU
        
        # Updates of total times
        tJs = tJs + tJ
        tUs = tUs + tU
        denum = denum+1
        
        # SNLLS1
        nDm1 = numData-1
        loss_value2,grads2,errs2 =  SNLLS1.gradJacA(models[1], x, y_one) # idxAbsChng,
        
        # Shuffle indices
        idxE = tf.random.shuffle(idxE)
        idxJ = tf.random.shuffle(idxJ)
        idxJs = idxJ[0:nDm1]
        
        ser = tf.reduce_sum(errs2)
        
        gk12 = tf.concat([tf.reshape(grads2[i],[-1]) for i in range(len(grads2))],axis=0)
        
        g1g1 = gk12*gk12
        gg = tf.reduce_sum(g1g1)
        
        ek = ek + ser*g1g1 # ser, np.abs(ser)
        
        #gDiag2 = gDiag2 + tf.math.abs(gk12) + rho*ek
        gDiag2 = gDiag2 + g1g1
        gDiag2Use = np.sqrt(gDiag2)
        #gDiag = 0.1*ovars + tf.math.abs(gk1) # No accumulation
        
        #beta = beta2/gDiag
        lk = lk + loss_value2
        jk,s2 = SNLLS1.update_step(models[1],jk,gk12,beta2/gDiag2Use,shps,lk,delta,delta1) # gDiag2
        
        # SNLLSL
        ts2 = time.time()
        loss_value3,grads3,errs,errs_unsort,idxAbsChng =  SNLLSL.gradJacA(models[2], x, y_one)
        
        gk1 = tf.concat([tf.reshape(grads3[i],[-1]) for i in range(len(grads3))],axis=0)
        
        gDiag3 = gDiag3 + gk1*gk1
        gDiag3Use = np.sqrt(gDiag3)
        #gDiag3 = gDiag3 + tf.math.abs(gk1)
            
        # Approximate Jacobian update
        # Using accumulation
        jacs3 = SNLLSL.update_stepA(models[2],grads3,jacs3,errs_unsort,numData,numVars,shps,(beta3/gDiag3Use),idxAbsChng) # gDiag3
        
        te2 = time.time()
        t2 = te2-ts2
        
        # model4 (SGD)
        loss_value4, grads4 = grad(models[3], x, y_one)
        optimizerSG.apply_gradients(zip(grads4, models[3].trainable_variables))
        
        # model5 (Adam)
        loss_value5, grads5 = grad(models[4], x, y_one)
        optimizerAD.apply_gradients(zip(grads5, models[4].trainable_variables))
        
        # model6 (Adagrad)
        loss_value6, grads6 = grad(models[5], x, y_one)
        optimizerAG.apply_gradients(zip(grads6, models[5].trainable_variables))
        #----------------------- End optimization ---------------------------------
        
        # Store losses
        losses[0] = losses[0] + loss_value
        losses[1] = losses[1] + loss_value2
        losses[2] = losses[2] + loss_value3
        losses[3] = losses[3] + loss_value4
        losses[4] = losses[4] + loss_value5
        losses[5] = losses[5] + loss_value6
        
        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y, models[0](x, training=True))
        #epoch_accuracy.update_state(y_one, model(x, training=True))
        
        # For model1
        epoch_loss_avg1.update_state(loss_value4)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy1.update_state(y, models[4](x, training=True))
    
      # End epoch
      train_loss_results.append(epoch_loss_avg.result())
      train_accuracy_results.append(epoch_accuracy.result())
      
      train_loss_results1.append(epoch_loss_avg1.result())
      train_accuracy_results1.append(epoch_accuracy1.result())
      
      # Timing updates
      times_ave_Jacs.append(tJs/denum)
      times_ave_Updates.append(tUs/denum)
    
      if epoch % 1 == 0: # 50
        print(("{:03d}   \t{:.4f}  {:.4f}  {:.4f}  {:.4f}"
               "  {:.4f}  {:.4f}").format(epoch,
                losses[0]/denum,
                losses[1]/denum,
                losses[2]/denum,
                losses[3]/denum,
                losses[4]/denum,
                losses[5]/denum))
        
        fileW.write(("{:03d},{:.10f},{:.10f},"
                     "{:.10f},{:.10f},{:.10f},{:.10f} ").format(epoch,
                 losses[0]/denum,
                 losses[1]/denum,
                 losses[2]/denum,
                 losses[3]/denum,
                 losses[4]/denum,
                 losses[5]/denum))

    fileW.close()


