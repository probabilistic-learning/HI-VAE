#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:56:13 2017

@author: anazabal, olmosUC3M, ivaleraM

List of loglikelihoods for the types of variables considered in this paper.
Basically, we create the different layers needed in the decoder and during the
generation of new samples

The variable reuse indicates the mode of this functions
- reuse = None -> Decoder implementation
- reuse = True -> Samples generator implementation

"""

import tensorflow as tf
import numpy as np

def loglik_real(batch_data,list_type,theta,normalization_params,tau2,kernel_initializer,name,reuse):
    
    output=dict()
    epsilon = tf.constant(1e-3, dtype=tf.float32)
    
    #Data outputs
    data, missing_mask = batch_data
    missing_mask = tf.cast(missing_mask,tf.float32)
    
    data_mean, data_var = normalization_params
    data_var = tf.clip_by_value(data_var, epsilon, np.inf)
    
    est_mean, est_var = theta
    est_var = tf.clip_by_value(tf.nn.softplus(est_var), epsilon, 1e20) #Must be positive
    
    # Affine transformation of the parameters
    est_mean = tf.sqrt(data_var)*est_mean + data_mean
    est_var = data_var*est_var
#    est_var = 0.05*tf.ones_like(est_var)
    
    #Compute loglik
    log_p_x = -0.5 * tf.reduce_sum(tf.squared_difference(data,est_mean)/est_var,1) - int(list_type['dim'])*0.5*tf.log(2*np.pi) - 0.5*tf.reduce_sum(tf.log(est_var),1)
#    log_p_x = -0.5 * tf.reduce_sum(tf.squared_difference(data,est_mean),1)
    
    #Outputs
    output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
    output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
    output['params'] = [est_mean, est_var]
    output['samples'] = tf.contrib.distributions.Normal(est_mean,tf.sqrt(est_var)).sample()
        
    return output


def loglik_pos(batch_data,list_type,theta,normalization_params,tau2,kernel_initializer,name,reuse):
    
    #Log-normal distribution
    output = dict()
    epsilon = tf.constant(1e-3, dtype=tf.float32)
    
    #Data outputs
    data_mean_log, data_var_log = normalization_params
    data_var_log = tf.clip_by_value(data_var_log, epsilon, np.inf)
    
    data, missing_mask = batch_data
    data_log = tf.log(1.0 + data)
    missing_mask = tf.cast(missing_mask,tf.float32)
    
    est_mean, est_var = theta
    est_var = tf.clip_by_value(tf.nn.softplus(est_var), epsilon, 1.0)
    
    # Affine transformation of the parameters
    est_mean = tf.sqrt(data_var_log)*est_mean + data_mean_log
    est_var = data_var_log*est_var
    

    #Compute loglik
    log_p_x = -0.5 * tf.reduce_sum(tf.squared_difference(data_log,est_mean)/est_var,1) \
        - 0.5*tf.reduce_sum(tf.log(2*np.pi*est_var),1) - tf.reduce_sum(data_log,1)
    
    output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
    output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
    output['params'] = [est_mean, est_var]
    output['samples'] = tf.clip_by_value(tf.exp(tf.contrib.distributions.Normal(est_mean,tf.sqrt(est_var)).sample()) - 1.0,0,1e20)
        
    return output

def loglik_cat(batch_data,list_type,theta,normalization_params,tau2,kernel_initializer,name,reuse):
    
    output=dict()
    
    #Data outputs
    data, missing_mask = batch_data
    missing_mask = tf.cast(missing_mask,tf.float32)
    
    log_pi = theta
    
    #Compute loglik
    log_p_x = -tf.nn.softmax_cross_entropy_with_logits(logits=log_pi,labels=data)
    
    output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
    output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
    output['params'] = log_pi
    output['samples'] = tf.one_hot(tf.contrib.distributions.Categorical(probs=tf.nn.softmax(log_pi)).sample(),depth=int(list_type['dim']))
    
    return output
    
def loglik_ordinal(batch_data,list_type,theta,normalization_params,tau2,kernel_initializer,name,reuse):
    
    output=dict()
    epsilon = tf.constant(1e-6, dtype=tf.float32)
    
    #Data outputs
    data, missing_mask = batch_data
    missing_mask = tf.cast(missing_mask,tf.float32)
    batch_size = tf.shape(data)[0]
    
    #We need to force that the outputs of the network increase with the categories
    partition_param, mean_param = theta
    mean_value = tf.reshape(mean_param,[-1,1])
    theta_values = tf.cumsum(tf.clip_by_value(tf.nn.softplus(partition_param), epsilon, 1e20),1)
    sigmoid_est_mean = tf.nn.sigmoid(theta_values - mean_value)
    mean_probs = tf.concat([sigmoid_est_mean,tf.ones([batch_size,1],tf.float32)],1) - tf.concat([tf.zeros([batch_size,1],tf.float32),sigmoid_est_mean],1)
    
    mean_probs = tf.clip_by_value(mean_probs,epsilon,1.0)
    
    #Code needed to compute samples from an ordinal distribution
    true_values = tf.one_hot(tf.reduce_sum(tf.cast(data,tf.int32),1)-1,int(list_type['dim']))
    
    #Compute loglik
    log_p_x = -tf.nn.softmax_cross_entropy_with_logits(logits=tf.log(mean_probs),labels=true_values)
    
    output['log_p_x'] = tf.multiply(log_p_x,missing_mask)
    output['log_p_x_missing'] = tf.multiply(log_p_x,1.0-missing_mask)
    output['params'] = mean_probs
    output['samples'] = tf.sequence_mask(1+tf.contrib.distributions.Categorical(logits=tf.log(tf.clip_by_value(mean_probs,epsilon,1e20))).sample(), int(list_type['dim']),dtype=tf.float32)
    
    return output
    
def loglik_count(batch_data,list_type,theta,normalization_params,tau2,kernel_initializer,name,reuse):
    
    output=dict()
    epsilon = tf.constant(1e-6, dtype=tf.float32)
    
    #Data outputs
    data, missing_mask = batch_data
    missing_mask = tf.cast(missing_mask,tf.float32)
    
    est_lambda = theta
    est_lambda = tf.clip_by_value(tf.nn.softplus(est_lambda),epsilon,1e20)
    
    log_p_x = -tf.reduce_sum(tf.nn.log_poisson_loss(targets=data,log_input=tf.log(est_lambda),compute_full_loss=True),1)
    
    output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
    output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
    output['params'] = est_lambda
    output['samples'] = tf.contrib.distributions.Poisson(est_lambda).sample()
        
    return output