#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:04:40 2021

@author: XX
"""

import tensorflow as tf
import numpy as np

### NLLS
def gradJac(model, inputs, targets, batch_size_Jac):
  with tf.GradientTape(persistent=True) as tape:      
      # Function "gradJac" computes the gradient and subsampled Jacobian of 
      # the loss_value. If "batch_size_Jac" == len(errs) then the full
      # Jacobian is computed
#      loss_value = loss(model, inputs, targets, training=True)
      predictions = model(inputs, training=True)
      errsF = predictions - targets
      errs = tf.reshape(errsF, [-1])
      
      loss_value = tf.reduce_sum(errs*errs)/len(errs)

      cands = tf.random.uniform_candidate_sampler(np.zeros((1,1)),1,batch_size_Jac,True,len(errs))

      idxJac = cands.sampled_candidates
      selErrs = tf.gather(errs,idxJac)
      
  return loss_value, tape.gradient(loss_value, model.trainable_variables), errs, tape.jacobian(selErrs, model.trainable_variables, experimental_use_pfor=False), idxJac

# Update step function
def update_step(model,grads,jacs,numData,numVars,shps):
    #------------ Function to compute a search direction ----------------------
    # Compute a "stacked" gradient
    numLays = len(grads)
    gradS = -tf.concat([tf.reshape(grads[i],[-1]) for i in range(numLays)],axis=0)    
    # Compute a "stacked" Jacobian (in trasposed form)
    jacSt = tf.concat([tf.reshape(jacs[i],[numData, -1]) for i in range(numLays)],axis=1) # jacV
    
    # Modified Algorithm
    if numVars < 50:
        # Defining and solving the linear system
        In = np.eye(numVars)
        B = In + tf.transpose(jacSt)@jacSt
        # Solve for trial step
        st = np.linalg.solve(B,gradS)
    else:
        # Sherman-Morrison-Woodbury formula (requires numData < numVars)
        st = gradS
        p = tf.linalg.matvec(jacSt,gradS,a_is_sparse=True,b_is_sparse=True)
        #p = tf.linalg.matvec(jacSt,gradS,a_is_sparse=False,b_is_sparse=False)
        #p = jacSt@gradS
        In = np.eye(numData)
        #B = In + jacSt@tf.transpose(jacSt)
        B = In + tf.linalg.matmul(jacSt,jacSt,transpose_b=True,a_is_sparse=True,b_is_sparse=True)
        #B = In + tf.linalg.matmul(jacSt,jacSt,transpose_b=True,a_is_sparse=False,b_is_sparse=False)
        p = np.linalg.solve(B,p)
        #st = st + tf.transpose(jacSt)@p
        #st = st - tf.linalg.matvec(jacSt,p,transpose_a=True,a_is_sparse=False,b_is_sparse=False)
        st = st - tf.linalg.matvec(jacSt,p,transpose_a=True,a_is_sparse=True,b_is_sparse=False)
        
        # For comparison
        #B1 = np.eye(numVars) + tf.transpose(jacSt)@jacSt
        # Solve for trial step
        #st1 = np.linalg.solve(B1,gradS)        
        #errs = st-st1
    
    # Defining and solving the linear system
    #B = In + tf.transpose(jacSt)@jacSt
    # Solve for trial step
    #st = np.linalg.solve(B,gradS)
    
    # Update variables
    idxS = 0
    idxE = 0
    for i in range(numLays):
        shpi = shps[i]
        lgs = np.prod(shpi)
        idxE = idxE + lgs
        # Updates
        model.trainable_variables[i].assign_add(tf.reshape(st[idxS:idxE],shpi))
        idxS = idxE