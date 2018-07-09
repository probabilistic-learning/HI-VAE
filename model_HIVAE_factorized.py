#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:23:35 2018

@author: anazabal, olmosUC3M, ivaleraM
"""

# # -*- coding: utf-8 -*-

## 2 fully connected layers at both encoding and decoding
# hidden_dim is the number of neurons of the first hidden layer

import tensorflow as tf
import VAE_functions
        
def encoder(X_list, miss_list, batch_size, z_dim, s_dim, tau):
    
    samples = dict.fromkeys(['s','z','y','x'],[])
    q_params = dict()
    X = tf.concat(X_list,1)
    
    #Create the proposal of q(s|x^o)
    samples['s'], q_params['s'] = VAE_functions.s_proposal_multinomial(X, batch_size, s_dim, tau, reuse=None)
    
    #Create the proposal of q(z|s,x^o)
    samples['z'], q_params['z'] = VAE_functions.z_proposal_GMM_factorized(X_list, samples['s'], miss_list, batch_size, z_dim, reuse=None)
    
    return samples, q_params
        

def decoder(batch_data_list, miss_list, types_list, samples, q_params, normalization_params, batch_size, z_dim, y_dim, y_dim_partition):
    
    p_params = dict()
    
    #Create the distribution of p(z|s)
    p_params['z'] = VAE_functions.z_distribution_GMM(samples['s'], z_dim, reuse=None)
    
    #Create deterministic layer y
    samples['y'] = tf.layers.dense(inputs=samples['z'], units=y_dim, activation=None,
                         kernel_initializer=tf.random_normal_initializer(stddev=0.05), name= 'layer_h1_', reuse=None)
    
    grouped_samples_y = VAE_functions.y_partition(samples['y'], types_list, y_dim_partition)

    #Compute the parameters h_y
    theta = VAE_functions.theta_estimation_from_y(grouped_samples_y, types_list, miss_list, batch_size, reuse=None)
    
    #Compute loglik and output of the VAE
    log_p_x, log_p_x_missing, samples['x'], p_params['x'] = VAE_functions.loglik_evaluation(batch_data_list, types_list, miss_list, theta, normalization_params, reuse=None)
        
    return theta, samples, p_params, log_p_x, log_p_x_missing

def cost_function(log_p_x, p_params, q_params, types_list, z_dim, y_dim, s_dim):
    
    #KL(q(s|x)|p(s))
    log_pi = q_params['s']
    pi_param = tf.nn.softmax(log_pi)
    KL_s = -tf.nn.softmax_cross_entropy_with_logits(logits=log_pi, labels=pi_param) + tf.log(float(s_dim))
    
    #KL(q(z|s,x)|p(z|s))
    mean_pz, log_var_pz = p_params['z']
    mean_qz, log_var_qz = q_params['z']
    KL_z = -0.5*z_dim +0.5*tf.reduce_sum(tf.exp(log_var_qz - log_var_pz) +tf.square(mean_pz - mean_qz)/tf.exp(log_var_pz) -log_var_qz + log_var_pz,1)
    
    #Eq[log_p(x|y)]
    loss_reconstruction = tf.reduce_sum(log_p_x,0)
    
    #Complete ELBO
    ELBO = tf.reduce_mean(loss_reconstruction - KL_z - KL_s,0)
    
    return ELBO, loss_reconstruction, KL_z, KL_s

def samples_generator(batch_data_list, X_list, miss_list, types_list, batch_size, z_dim, y_dim, y_dim_partition, s_dim, tau, normalization_params):
    
    samples_test = dict.fromkeys(['s','z','y','x'],[])
    test_params = dict()
    X = tf.concat(X_list,1)
    
    #Create the proposal of q(s|x^o)
    _, params = VAE_functions.s_proposal_multinomial(X, batch_size, s_dim, tau, reuse=True)
    samples_test['s'] = tf.one_hot(tf.argmax(params,1),depth=s_dim)
    
    #Create the proposal of q(z|s,x^o)
    _, params = VAE_functions.z_proposal_GMM_factorized(X_list, samples_test['s'], miss_list, batch_size, z_dim, reuse=True)
    samples_test['z'] = params[0]
    
    #Create deterministic layer y
    samples_test['y'] = tf.layers.dense(inputs=samples_test['z'], units=y_dim, activation=None,
                         kernel_initializer=tf.random_normal_initializer(stddev=0.05), trainable=True, name= 'layer_h1_', reuse=True)
    
    grouped_samples_y = VAE_functions.y_partition(samples_test['y'], types_list, y_dim_partition)
    
    #Compute the parameters h_y
    theta = VAE_functions.theta_estimation_from_y(grouped_samples_y, types_list, miss_list, batch_size, reuse=True)
    
    #Compute loglik and output of the VAE
    log_p_x, log_p_x_missing, samples_test['x'], test_params['x'] = VAE_functions.loglik_evaluation(batch_data_list, types_list, miss_list, theta, normalization_params, reuse=True)
    
    return samples_test, test_params, log_p_x, log_p_x_missing