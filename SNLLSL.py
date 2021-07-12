#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNLLSL.py
Script implementing the functions for SNLLSL

@author: XX
"""

import tensorflow as tf
import numpy as np

def gradJacA(model, inputs, targets): # , errs_prev
  with tf.GradientTape(persistent=True) as tape:
      predictions = model(inputs, training=True)
      
      errs = (tf.reshape(predictions,[-1]) - tf.reshape(targets,[-1]))
      
      loss_value = tf.reduce_sum(errs*errs)/len(errs)
      
      # Random selection
      cands = tf.random.uniform_candidate_sampler(np.zeros((1,1)),1,len(errs),True,len(errs))
      idxErrsOut = cands.sampled_candidates

  return loss_value, tape.gradient(loss_value, model.trainable_variables), errs,  errs, idxErrsOut #errs, idxAbsChng

# Update step function (including parameter "beta") for the approximate 
# Jacobian method
def update_stepA(model,grads,jacs_in,errs,numData,numVars,shps,beta,idxAbsChng):
    #------------ Function to compute a search direction ----------------------
    # Compute a "stacked" gradient
    numLays = len(grads)
    gradS = tf.concat([tf.reshape(grads[i],[-1]) for i in range(numLays)],axis=0)    
    
    nDm1 = len(idxAbsChng)-1
    #nDm1 = numData-1
    
    #nerr = 0
#    valTG,idxG = tf.math.top_k(tf.abs(gradS),k=nDm1)
    
    # Random selection
    cands = tf.random.uniform_candidate_sampler(np.zeros((1,1)),1,nDm1,True,len(gradS))
    idxG = cands.sampled_candidates
    
    #idxG = tf.argsort(tf.abs(gradS),direction='DESCENDING')
    
    # Safeguarding
    errs = tf.where(tf.abs(errs)>0.00001,errs,0.00001)
    
    # Setting up approximated Jacobian
    denom = errs[idxAbsChng[nDm1]]

    # Safeguarding            
#    if tf.abs(denom) < 0.001:
#        denom = tf.sign(denom)*0.001
    
    # Update approximate Jacobian
    # Prepare required tensors
    jacs_local = gradS/denom
    gradIdx = tf.gather(gradS,idxG)
    errsIdx = tf.gather(errs,idxAbsChng[0:nDm1])
    idxGR = tf.reshape(idxG,[len(idxG),1])
    jacUpdate = gradIdx/(errsIdx)
    
    # Update Jacobians
    jacs = tf.tensor_scatter_nd_update(jacs_local,idxGR,jacUpdate)
    
    # Accumulate Jacobians
    jacs = jacs_in + jacs
    
    # Compute a "stacked" Jacobian (in trasposed form)
    #jacSt = tf.concat([tf.reshape(jacV[i],[numData, -1]) for i in range(numLays)],axis=1)
    
    # Modified Algorithm
    if numVars < 50:
        # Defining and solving the linear system
        In = np.eye(numVars)
        B = (1/beta)*In + tf.transpose(jacSt)@jacSt
        # Solve for trial step
        st = np.linalg.solve(B,gradS)
    else:
        # Sherman-Morrison-Woodbury formula (requires numData < numVars)
        st = (-beta)*gradS
        jst = jacs*st
        jj = jacs*(beta*jacs)
        
        sumJst = tf.reduce_sum(jst[nDm1:numVars],keepdims=True)
        sumJJ = tf.reduce_sum(jj[nDm1:numVars],keepdims=True)
        #p = (tf.linalg.matvec(jacSt,st,a_is_sparse=True,b_is_sparse=True))
        p = tf.concat([jst[0:nDm1],sumJst],axis=0) #(tf.linalg.matvec(jacSt,st,a_is_sparse=True,b_is_sparse=True))
        p1 = tf.concat([jj[0:nDm1],sumJJ],axis=0)
        
        p2 = p/(1+p1) # numData tf.sqrt(numData/2), (numData/2), 10 ,20
        
        jp2a = (jacs[0:nDm1]*p2[0:nDm1])
        jp2b = (jacs[nDm1:numVars]*p2[nDm1])
        jp2 = -beta*tf.concat([jp2a,jp2b],axis=0)
        
        st = st + jp2
        
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
     
    # Method accumulates jacobians    
    return jacs