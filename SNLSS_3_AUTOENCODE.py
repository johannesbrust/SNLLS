#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SNLLS_3_AUTOENCODE.py
#------------------------------------------------------------------------------
# Script to test 5 algorithms on a least-squares classification problem.
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
#        SNLLS1 (rank-1 stochastic jacobian least-squares)
#        SNLLSL (rank-L stochastic jacobian least-squares)
#        SGD 
#        ADAM 
#        ADAGRAD
#------------------------------------------------------------------------------
    MODIFICATIONS:
        03/04/21, X.X., Use of only the 1st example
        06/07/21, X.X., Implementation of training loop using tf.data.Dataset
        06/09/21, X.X., Use of smaller data
        06/10/21, X.X., Measuring timings, add a shuffle operation
        outside of step computations/derivatives
        06/11/21, X.X., Measuring step lengths and scaling parameter
        06/16/21, X.X., Further timing of method
                        Implementation of method with rank-1 second derivative
                        estimate
        06/17/21, X.X., Further testing
            Implementation of Jacobian accumulation approach
        06/18/21, X.X., Preparing estimates for 2nd derivative matrix
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

#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import tensorflow as tf

# Additional imports
import time

#from sklearn.metrics import accuracy_score, precision_score, recall_score
#from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# Solver imports
### Solver imports
#import NLLS
import SNLLS1
import SNLLSL

# Setup for storing data
dataPath = './DATA/'
filePrefix = 'EX_3'
numRuns = 5 # 10
# Set epochs throughout the experiment
numEpochs = 25
#numRuns = 1

## Load dataset

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)

# Setup model

#latent_dim = 32
latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
autoencoder = Autoencoder(latent_dim)

optSGD = tf.keras.optimizers.SGD(learning_rate=20)
#autoencoder.compile(optimizer=optSGD, loss=losses.MeanSquaredError()) #adam adagrad sgd
#autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError()) #adam adagrad sgd

# Second model
model = Autoencoder(latent_dim)

# Number of algorithms in this experiment
# Algs. 1--2 correspond to NLLS methods, and 3--5 correspond to known methods
numAlgs = 5

# Initialize one model per algorithm
models = []
for i in range(numAlgs):
    models.append(Autoencoder(latent_dim))


# Defining MeanSquaredError loss and data
loss_object = tf.keras.losses.MeanSquaredError()
batch_size = 32
batchIndex = 0
buffer_size = 60000 # the training data 32*1875=60000
#buffer_size = 10000
train_data = tf.data.Dataset.from_tensor_slices((x_train,x_train))
#train_data = tf.data.Dataset.from_tensors((x_test,x_test))
train_data = train_data.shuffle(buffer_size=buffer_size).batch(batch_size)

n_iter = next(iter(train_data))
features = n_iter[0]
labels = n_iter[1]

# Defining loss

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  return loss_object(y_true=y, y_pred=y_)

# Initialize models    
# Assign same initial variables to all models and
# print losses 
lossesL = [] #np.zeros((numAlgs,1))
for m in range(numAlgs):
    predictions = models[m](features)
    for i in range(len(models[m].trainable_variables)):
        models[m].trainable_variables[i].assign(models[0].trainable_variables[i])
        
    lossesL.append(loss(models[m],features,labels,training=True))

print("\nLoss test (Initial): \n")
 
solverFormat = "SNLLS1  SNLLSL  SGD     ADAM    ADAGRAD \n"
lossFormat = ("{0[0]:.4f}  {0[1]:.4f}  {0[2]:.4f}"
                "  {0[3]:.4f}  {0[4]:.4f} \n")

print("Epoch \t"+solverFormat)
print("Init. \t"+lossFormat.format(lossesL))

# Gradient to define the training algorithm
# The nonlinear least-squares methods will have additional
# gradient computation functions, because Jacobians (or --
#     stochastic approximations) are also included
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Call to the approximate Jacobian function (for testing)
# Test call to ensure all is fine. And to determine problem size
loss_valueA,gradsA,errs =  SNLLS1.gradJacA(model, features, labels) # idxAbsChng,

# Setup for step computation
k = 0
numSolv = 0  

numData = len(errs)
numLays = len(gradsA)
varDims = np.zeros((numLays,2))
numVars = 0
shps = []
for i in range(numLays):
    shpi = gradsA[i].shape
    shps.append(shpi)
    #varDims[i,0] = shpLay[0]
    #varDims[i,1] = shpLay[1]
    numVars = numVars + np.prod(shpi)
    
# Print size of variables and Data (for batch)
print("\nSize variables: {}, Size data: {}".format(numVars,
                                          numData))    

## KNOWN OPTIMIZER
learning_rate = 50.0        
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
delta = 0.9 # 0.05 np.sqrt(numData/(2*(numSamples/batchSize)))
delta1 = 1.0

rho = 0.0005 
ek = np.zeros(numVars)

gDiag2 = 0.0000000001 + tf.zeros(numVars) #0.1*tf.ones(numVars)

beta2 = 0.05 #1 7 0.1

# SNLLSL (model3)
beta3 = 0.05 

#betaDiag = beta*tf.ones(numVars)
gDiag3 = 0.0000000001 + tf.zeros(numVars) # 0.1*tf.ones(numVars)
jacs3 = tf.Variable(np.zeros(numVars),dtype=tf.float32)

# Intermediate variables
jacs = tf.Variable(np.zeros(numVars),dtype=tf.float32)
zJacs = tf.Variable(np.zeros(numVars),dtype=tf.float32)
    
wk = tf.concat([tf.reshape(model.trainable_variables[i],[-1]) for i in range(len(model.trainable_variables))],axis=0)
gk = tf.concat([tf.reshape(gradsA[i],[-1]) for i in range(len(gradsA))],axis=0)    

# Intermediate storage
train_loss_results = []
times_ave_Jacs = []
times_ave_Updates = []
times_ave_internal_Jacs = []
times_ave_internal_Updates = []
times_ave_shuffle = []
losses = []
sumErrs = []

# Storing scaling sizes and norms of steps
norms_steps_ave = []
scalings_ave = []
scalings_ave1 = []

# Storing every loss and every "scaling 1"
lossesAll = []
scalings1All = []
sumErrsAll = []
sumFactorsAll = []

k = 0
numSolv = 1
num_epochs = numEpochs # 5 # 5 15 25 50

initVal = 1/(loss_valueA*loss_valueA)

beta_ = 7 #0.00001# 20 # 0.8 #learning_rate # 0.5 0.8 #learning_rate #1#learning_rate
gamma = 0.001 # 0.00001 # Additional parameter

ovars = tf.ones(numVars)
betaDiag = beta_*ovars
gDiag = 0.1*tf.ones(numVars)
yDiag = 0.05*tf.ones(numVars)
syDiag = 0.05*tf.ones(numVars)#tf.zeros(numVars)


sek = 0.0

# Possible adjustment to delta
sumFact = 0.0

## Training loop

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
            model = Autoencoder(latent_dim)
            models.append(model)
            predictions = models[i](features)
            for j in range(len(models[i].trainable_variables)):
                models[i].trainable_variables[j].assign(models[0].trainable_variables[j])
        
        idxE = tf.constant(range(numData0))
        idxJ = tf.constant(range(numVars))
        nDm1 = numData0-1
        
        # Variables being accumulated
        jk = tf.Variable(np.zeros(numVars),dtype=tf.float32)
        lk = 0
        ek = np.zeros(numVars)        
        gDiag2 = 0.0000000001 + tf.zeros(numVars)
        gDiag3 = 0.0000000001 + tf.zeros(numVars)
        #gDiag2 = tf.ones(numVars)
        #gDiag3 = tf.ones(numVars)
        jacs3 = tf.Variable(np.zeros(numVars),dtype=tf.float32)
        
        # Method has additional parameters
        #wk = tf.concat([tf.reshape(models[1].trainable_variables[i],[-1]) for i in range(len(models[1].trainable_variables))],axis=0)
        #gk = tf.concat([tf.reshape(grads[i],[-1]) for i in range(len(grads))],axis=0)

    # Training loop
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
      
      # Training loop 
      # Timing data
      timesJ = tf.zeros(4)
      timesS = tf.zeros(8)
      tJs = 0.0
      tUs = 0.0
      tSU = 0.0
      denum = 0
      normSteps = tf.zeros(2)
      scaling = 0.0
      scaling1 = 0.0
      local_loss = 0.0;
      
      se = 0.0
      
      # Reinitialize scaling
      #yDiag = 0.05*tf.ones(numVars)
      #jacs = tf.zeros(numVars)
      #jacs = tf.Variable(np.zeros(numVars),dtype=tf.float32)
      
      for step, (features, labels) in enumerate(train_data):
     #di in cached_train:
        # Convert labels  
        #y_one = tf.one_hot(y,3)  
        #numData = np.prod(y_one.shape)
        
        #features = (di["user_id"],di["movie_title"])
        #labels = di["user_rating"]
    
        numData = len(labels)
    
        #---------------------- Optimize the model--------------------------------
        # This uses the "nonlinear least squares (NLLS)" update_step function
        # and gradient and Jacobian computations
        # This uses an approximate Jacobian
            
        tsJ = time.time()
        
        # Approximate Jacobian for legacy references
        loss_value,grads,errs =  SNLLS1.gradJacA(model, features, labels) # idxAbsChng,
                
        ## Solvers
        # SNLLS1
        #nDm1 = numData-1
        loss_value2,grads2,errs2 =  SNLLS1.gradJacA(models[0], features,
                                                    labels) # idxAbsChng,
        
        loss_value2, grads2 = grad(models[0], features, labels)
        
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
        
        gDiag2 = gDiag2 + g1g1 # + rho*ek
        gDiag2U = np.sqrt(gDiag2)
        #gDiag = 0.1*ovars + tf.math.abs(gk1) # No accumulation
        
        #beta = beta2/gDiag
        lk = lk + loss_value2
        jk,s2 = SNLLS1.update_step(models[0],jk,gk12,beta2/gDiag2U,shps,lk,delta,delta1)
        
        # SNLLSL
        ts2 = time.time()
        loss_value3,grads3,errs,errs_unsort,idxAbsChng =  SNLLSL.gradJacA(models[1],
                                                                          features,
                                                                          labels)
        
        loss_value3, grads3 = grad(models[1], features, labels)
        
        gk1 = tf.concat([tf.reshape(grads3[i],[-1]) for i in range(len(grads3))],axis=0)
        
        #gDiag3 = gDiag3 + tf.math.abs(gk1)
        gDiag3 = gDiag3 + gk1*gk1
        gDiag3U = np.sqrt(gDiag3)
        
        # Approximate Jacobian update
        # Using accumulation
        jacs3 = SNLLSL.update_stepA(models[1],grads3,jacs3,errs_unsort,numData,numVars,shps,(beta3/gDiag3U),idxAbsChng)
        
        # Additional parameter 
        
        te2 = time.time()
        t2 = te2-ts2
        
        # model4 (SGD)
        loss_value4, grads4 = grad(models[2], features, labels)
        optimizerSG.apply_gradients(zip(grads4, models[2].trainable_variables))
        
        # model5 (Adam)
        loss_value5, grads5 = grad(models[3], features, labels)
        optimizerAD.apply_gradients(zip(grads5, models[3].trainable_variables))
        
        # model6 (Adagrad)
        loss_value6, grads6 = grad(models[4], features, labels)
        optimizerAG.apply_gradients(zip(grads6, models[4].trainable_variables))
        #----------------------- End optimization ---------------------------------
        
        # Store losses
        losses[0] = losses[0] + loss_value2
        losses[1] = losses[1] + loss_value3
        losses[2] = losses[2] + loss_value4
        losses[3] = losses[3] + loss_value5
        losses[4] = losses[4] + loss_value6
        
        # Legacy timings
        timingsS = np.zeros(8) 
        ts = time.time()
        te = time.time()
        timingsS[0] = te-ts        
        ts = time.time()
        te = time.time()
        timingsS[1] = te-ts
        ts = time.time()
        te = time.time()
        timingsS[2] = te-ts
        
        ts = time.time()        
        te = time.time()
        
        timingsS[3] = te-ts
        
        ts = time.time()        
        te = time.time()
        timingsS[4] = te-ts
        
        ts = time.time()
        te = time.time()
        timingsS[5]        
        ts = time.time()
            
        normSteps1 = np.zeros(2)

        te = time.time()
        timingsS[6] = te - ts
        
        ts = time.time()
                 
        te = time.time()
        timingsS[7] = te-ts

        numSolv = numSolv + 1
        teU = time.time()

        denum = denum+1
        
        timesJ = timesJ #+ timingsJ
        timesS = timesS + timingsS
        
        normSteps = normSteps + normSteps1
        
        local_loss = local_loss + loss_value
            
        se = se + ser
        sek = sek + ser
        
        lossesAll.append(loss_value)
#        scalings1All.append(scale1)
        sumErrsAll.append(sek)

        #----------------------- End optimization ---------------------------------
        
        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        #epoch_accuracy.update_state(y, model(x, training=True))
        #epoch_accuracy.update_state(y_one, model(x, training=True))
        
        # For model1
        #epoch_loss_avg1.update_state(loss_value1)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        #epoch_accuracy1.update_state(y, model1(x, training=True))
    
      # End epoch
      
      # Restart 
      #jacs = tf.Variable(np.zeros(numVars),dtype=tf.float32)
      errs_prev = tf.zeros(batch_size)
      
      train_loss_results.append(epoch_loss_avg.result())
      #train_accuracy_results.append(epoch_accuracy.result())
      
      #train_loss_results1.append(epoch_loss_avg1.result())
      #train_accuracy_results1.append(epoch_accuracy1.result())
      
      # Timing updates
      times_ave_Jacs.append(tJs/denum)
      times_ave_Updates.append(tUs/denum)
      times_ave_shuffle.append(tSU/denum)
      times_ave_internal_Jacs.append(timesJ/denum)
      times_ave_internal_Updates.append(timesS/denum)
      norms_steps_ave.append(normSteps/denum)
      scalings_ave.append(scaling/denum)
      scalings_ave1.append(scaling1/denum)
      #losses.append(local_loss/denum)
      sumErrs.append(se/denum)
      
      
      totalTime = (tJs/denum)+(tUs/denum)
      
      if epoch % 1 == 0: # 50
          print(("{:03d}   \t{:.4f}  {:.4f}  {:.4f}  {:.4f}"
               "  {:.4f}").format(epoch,
                losses[0]/denum,
                losses[1]/denum,
                losses[2]/denum,
                losses[3]/denum,
                losses[4]/denum))
                
        
          fileW.write(("{:03d},{:.10f},{:.10f},"
                     "{:.10f},{:.10f},{:.10f} ").format(epoch,
                 losses[0]/denum,
                 losses[1]/denum,
                 losses[2]/denum,
                 losses[3]/denum,
                 losses[4]/denum))
    

    fileW.close()
