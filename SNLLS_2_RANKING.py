#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SNLLS_2_RANKING.py
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
        01/18/21, X.X., Testing extracting Jacobians in a ML model
        03/12/21, X.X., Modifying this script for NLLS solver
        03/15/21, X.X., Modification to use "stochastic Jacobian"
        03/19/21, X.X., Comparing NLLS and "build-in" solver 
        (addition of scaling parameter "beta")
        03/22/21, X.X., Sorted errors
        04/05/21, X.X., Larger number of epochs
        04/15/21, X.X., Approach to approximate Jacobians 
        (instead of explict computation)
        04/16/21, X.X., Approximate Jacobian approach
        Accumulating Jacobian vectors
        06/21/21, X.X., Printing of results
        06/24/21, X.X., Rerun of experiment with "original" gradient
        computation
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

import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'tensorflow-recommenders'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'tensorflow-datasets'])    
    
#!pip install -q tensorflow-recommenders
#!pip install -q --upgrade tensorflow-datasets

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

# Additional imports
import time

### Solver imports
#import NLLS
import SNLLS1
import SNLLSL

# Setup for storing data
dataPath = './DATA/'
filePrefix = 'EX_2'
numRuns = 5 # 10
num_epochs = 50 # 15

# Setup of dataset

ratings = tfds.load("movielens/100k-ratings", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

## Model

class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])
    
  def call(self, inputs):

    user_id, movie_title = inputs

    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

# Testing the model

RankingModel()((["42"], ["One Flew Over the Cuckoo's Nest (1975)"]))

# Loss

task = tfrs.tasks.Ranking(
  loss = tf.keras.losses.MeanSquaredError(),
  metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# Prediction model

class MovielensModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    rating_predictions = self.ranking_model(
        (features["user_id"], features["movie_title"]))

    # The task computes the loss and the metrics.
    return self.task(labels=features["user_rating"], predictions=rating_predictions)

# Setup of experiment

# Number of algorithms in this experiment
# Algs. 1--2 correspond to NLLS methods, and 3--5 correspond to known methods
numAlgs = 5

# Initialize one model per algorithm
models = []
for i in range(numAlgs):
    models.append(MovielensModel())


# Warning: For large batch size the program may crash
batch_size = 8192 #4096 #8192 # 32, 8192

cached_train = train.shuffle(100_000).batch(batch_size).cache()
cached_test = test.batch(4096).cache()

# The underlying model is called via "model.ranking_model"
# The features are accessed via "user_id" and "movie_title"

# Definition of loss
loss_object = tf.keras.losses.MeanSquaredError()

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  return loss_object(y_true=y, y_pred=y_)

# Extract "features", "labels" (d1 stands for data 1)  
d1 = next(iter(cached_train))
features = (d1["user_id"],d1["movie_title"])
labels = d1["user_rating"]

# Initialize "models" by calling on it
for i in range(numAlgs):
    predictions = models[i].ranking_model(features)


# Assign same initial variables to all models and
# print losses 
lossesL = [] #np.zeros((numAlgs,1))
for m in range(numAlgs):
    for i in range(len(models[m].trainable_variables)):
        models[m].trainable_variables[i].assign(models[0].trainable_variables[i])
    
    lossesL.append(loss(models[m].ranking_model,features,labels,training=True))
        #losses[m] = loss(models[m],features,labels,training=True)
    
print("\nLoss test (Initial): \n")
 
solverFormat = "SNLLS1  SNLLSL  SGD     ADAM    ADAGRAD \n"
lossFormat = ("{0[0]:.4f}  {0[1]:.4f}  {0[2]:.4f}"
                "{0[3]:.4f}  {0[4]:.4f} \n")

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
# and for computing problem dimensions
loss_valueA,grads,errs,errs_chng,idxAbsChng =  SNLLSL.gradJacA(models[0].ranking_model, features, labels)

## Problem Size

numData = len(errs)
numLays = len(grads)
varDims = np.zeros((numLays,2))
numVars = 0
shps = []
for i in range(numLays):
    shpi = grads[i].shape
    shps.append(shpi)
    #varDims[i,0] = shpLay[0]
    #varDims[i,1] = shpLay[1]
    numVars = numVars + np.prod(shpi)
    
# Print size of variables and Data (for batch)
print("\nSize variables: {}, Size data: {}".format(numVars,
                                          numData))

## KNOWN OPTIMIZER
learning_rate = 1.0        
# SGD
optimizerSG = tf.keras.optimizers.SGD()#(learning_rate=0.01)

# ADAM
optimizerAD = tf.keras.optimizers.Adam()#(learning_rate=0.01)

# ADAGRAD
optimizerAG = tf.keras.optimizers.Adagrad(learning_rate=0.1)#(learning_rate=0.01)

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
delta = 20.0 # 0.05#0.00005 # np.sqrt(numData/(2*(numSamples/batchSize)))
delta1 = 1#0.85 # 0.001
delta1t = 1 # 0.85
delta2 = 1 # 0.999 #0.9
delta3 = 1 # 0.85

rho = 0 
ek = np.zeros(numVars)
gDiag2 = 0.0000000001 + tf.zeros(numVars) 

beta2 = 0.05 

beta21 = beta2
# SNLLSL (model3)
beta3 = 0.05 #learning_rate # 0.5 0.8 #learning_rate #1#learning_rate

#betaDiag = beta*tf.ones(numVars)
gDiag3 = 0.0000000001 + tf.zeros(numVars) # tf.ones(numVars)
jacs3 = tf.Variable(np.zeros(numVars),dtype=tf.float32)

# Method has additional parameters
wk = tf.concat([tf.reshape(models[1].trainable_variables[i],[-1]) for i in range(len(models[1].trainable_variables))],axis=0)
gk = tf.concat([tf.reshape(grads[i],[-1]) for i in range(len(grads))],axis=0)

wk2 = wk
gk2 = gk

jacs = tf.Variable(np.zeros(numVars),dtype=tf.float32)

# Intermediate variables
train_loss_results = []
times_ave_Jacs = []
times_ave_Updates = []
k = 0
numSolv = 1

## Start training loop

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
            model = MovielensModel()
            models.append(model)
            predictions = models[i].ranking_model(features)
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
        wk = tf.concat([tf.reshape(models[1].trainable_variables[i],[-1]) for i in range(len(models[1].trainable_variables))],axis=0)
        
        loss_value3,grads3,errs,errs_unsort,idxAbsChng =  SNLLSL.gradJacA(models[1].ranking_model,
                                                                          features,
                                                                          labels)
        
        gk = tf.concat([tf.reshape(grads3[i],[-1]) for i in range(len(grads3))],axis=0)

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
      
      #jk = tf.Variable(np.zeros(numVars),dtype=tf.float32)
      
      # Training loop 
      # Timing data
      tJs = 0.0
      tUs = 0.0
      denum = 0
      for di in cached_train:
        # Convert labels  
        #y_one = tf.one_hot(y,3)  
        #numData = np.prod(y_one.shape)
        
        features = (di["user_id"],di["movie_title"])
        labels = di["user_rating"]
    
        numData = len(labels)
    
        #---------------------- Optimize the model--------------------------------
        # This uses the "nonlinear least squares (NLLS)" update_step function
        # and gradient and Jacobian computations

        tsJ = time.time()
        
        # Update errors
        errs_prev = errs
        
        k = k + 1
        teJ = time.time()
        tJ = teJ-tsJ
        
        # Updated step
        tsU = time.time()
        
        ## Solvers
        # SNLLS1
        nDm1 = numData-1
        loss_value2,grads2,errs2 =  SNLLS1.gradJacA(models[0].ranking_model, features,
                                                    labels) # idxAbsChng,
        
        loss_value2, grads2 = grad(models[0].ranking_model, features, labels)
        
        # Shuffle indices
        idxE = tf.random.shuffle(idxE)
        idxJ = tf.random.shuffle(idxJ)
        idxJs = idxJ[0:nDm1]
        
        ser = tf.reduce_sum(errs2)
        
        gk12 = tf.concat([tf.reshape(grads2[i],[-1]) for i in range(len(grads2))],axis=0)
        
        delta1p = np.power(delta1,k)
        delta1tp = np.power(delta1t,k)
        delta2p = np.power(delta2,k)
        
        
        g1g1 = gk12*gk12
        gg = tf.reduce_sum(g1g1)
        ng = np.sqrt(gg)
        #delta1 = 0.0001 # 1/np.maximum(ng,1)
        
        ek = ek + ser*g1g1 # ser, np.abs(ser)
        
        #gDiag2 = gDiag2 + tf.math.abs(gk12) + rho*ek
        #gDiag2 = gDiag2 + tf.math.abs(gk12) + rho*ek
        gDiag2 = gDiag2 + tf.math.abs(g1g1) + rho*ek        
        
        #gDiag2 = delta2*gDiag2 + g1g1 + rho*ek
        
        #diagUse = beta2/gDiag2 # beta2/np.sqrt(gDiag2)
        diagUse = beta21/np.sqrt(gDiag2) # beta2/np.sqrt(gDiag2), beta2
#        diagUse = beta3/np.sqrt(gDiag2) # beta2/np.sqrt(gDiag2), beta2
        #gDiag = 0.1*ovars + tf.math.abs(gk1) # No accumulation
        
        #beta = beta2/gDiag
        lk = lk + delta3*loss_value2
        #delta = np.sqrt(lk)
        jk,s2 = SNLLS1.update_step(models[0],jk,gk12,diagUse,shps,lk,delta,delta1) # beta2 beta3 gDiag2
        
        #sk = wk1-wk
        yk2 = gk12-gk2
        
        #yy = tf.matmul(yk,yk)
        #sy = tf.matmul(yk,sk)
        yy2 = tf.reduce_sum(yk2*yk2)
        ss2 = tf.reduce_sum(s2*s2)
        sy2 = tf.reduce_sum(yk2*s2)
        
  #      wk = wk1
        gk2 = gk12
                
        # SNLLSL
        ts2 = time.time()
        loss_value3,grads3,errs,errs_unsort,idxAbsChng =  SNLLSL.gradJacA(models[1].ranking_model,
                                                                          features,
                                                                          labels)
        loss_value3, grads3 = grad(models[1].ranking_model, features, labels)
        
        gk1 = tf.concat([tf.reshape(grads3[i],[-1]) for i in range(len(grads3))],axis=0)
        
        #gDiag3 = gDiag3 + tf.math.abs(gk1)
        gDiag3 = gDiag3 + gk1*gk1
        
        
        gDiag3U = np.sqrt(gDiag3)
        # Approximate Jacobian update
        # Using accumulation
        jacs3 = SNLLSL.update_stepA(models[1],grads3,jacs3,errs_unsort,numData,numVars,shps,(beta3/gDiag3U),idxAbsChng) # gDiag3
        
        # New gvariables       
        wk1 = tf.concat([tf.reshape(models[1].trainable_variables[i],[-1]) for i in range(len(models[1].trainable_variables))],axis=0)
        
        sk = wk1-wk
        yk = gk1-gk
        
        #yy = tf.matmul(yk,yk)
        #sy = tf.matmul(yk,sk)
        yy = tf.reduce_sum(yk*yk)
        ss = tf.reduce_sum(sk*sk)
        sy = tf.reduce_sum(yk*sk)
                
        wk = wk1
        gk = gk1
        
        # Additional parameter 
        
        te2 = time.time()
        t2 = te2-ts2
        
        # model4 (SGD)
        loss_value4, grads4 = grad(models[2].ranking_model, features, labels)
        optimizerSG.apply_gradients(zip(grads4, models[2].trainable_variables))
        
        # model5 (Adam)
        loss_value5, grads5 = grad(models[3].ranking_model, features, labels)
        optimizerAD.apply_gradients(zip(grads5, models[3].trainable_variables))
        
        # model6 (Adagrad)
        loss_value6, grads6 = grad(models[4].ranking_model, features, labels)
        optimizerAG.apply_gradients(zip(grads6, models[4].trainable_variables))
        #----------------------- End optimization ---------------------------------
        
        # Store losses
        losses[0] = losses[0] + loss_value2
        losses[1] = losses[1] + loss_value3
        losses[2] = losses[2] + loss_value4
        losses[3] = losses[3] + loss_value5
        losses[4] = losses[4] + loss_value6
        #losses[5] = losses[5] + loss_value6
        
        numSolv = numSolv + 1
        teU = time.time()
        tU = teU-tsU
        
        # Updates of total times
        tJs = tJs + tJ
        tUs = tUs + tU
        denum = denum+1
        
        # model1
        #loss_value1, grads1 = grad(model1, x, y_one)
        #optimizer.apply_gradients(zip(grads1, model1.trainable_variables))
        #----------------------- End optimization ---------------------------------
        
        # Track progress
        epoch_loss_avg.update_state(loss_value2)  # Add current batch loss
    
      # End epoch
      
      # Restart 
      #jacs = tf.Variable(np.zeros(numVars),dtype=tf.float32)
      errs_prev = tf.zeros(batch_size)
      
      train_loss_results.append(epoch_loss_avg.result())
      
      # Timing updates
      times_ave_Jacs.append(tJs/denum)
      times_ave_Updates.append(tUs/denum)
      
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
    