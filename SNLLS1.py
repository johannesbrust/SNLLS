#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNLLSS1.py

Implementation of the stochastic nonlinear least squares solver with
a rank-1 Jacobian approximation

@author: XX
"""

import tensorflow as tf
import numpy as np

def gradJacA(model, inputs, targets):
  with tf.GradientTape(persistent=False) as tape: # True
            
      predictions = model(inputs, training=True)
      errsF = predictions - targets
      errs = tf.reshape(errsF, [-1])
      
      loss_value = tf.reduce_sum(errs*errs)/len(errs)
      
  return loss_value, tape.gradient(loss_value, model.trainable_variables), errs

def update_step(model,jk,gk1,beta,shps,lk,delta,delta1):
    jk = delta1*jk + gk1
    vk = delta*(1/np.sqrt(lk))*jk
    
    s1 = beta*(-gk1)
    aa1 = tf.reduce_sum(vk*s1)
    
    p1 = beta*vk
    aa2 = tf.reduce_sum(vk*p1)
    aa3 = aa1/(1+aa2)
    
    st = s1 - aa3*p1
    
    idxSs = 0
    idxEe = 0
    for i in range(len(shps)):
        shpi = shps[i]
        lgs = np.prod(shpi)
        idxEe = idxEe + lgs
        # Updates
        model.trainable_variables[i].assign_add(tf.reshape(st[idxSs:idxEe],shpi))
        idxSs = idxEe
        
    return jk, st