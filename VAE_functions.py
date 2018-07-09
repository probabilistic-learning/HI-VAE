#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:59:14 2018

@author: anazabal
"""

import csv
import tensorflow as tf
import loglik_models_missing_normalize
import numpy as np

def place_holder_types(types_file, batch_size):
    
    #Read the types of the data from the files
    with open(types_file) as f:
        types_list = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
        
    #Create placeholders for every data type, with appropriate dimensions
    batch_data_list = []
    for i in range(len(types_list)):
        batch_data_list.append(tf.placeholder(tf.float32, shape=(batch_size,types_list[i]['dim'])))
    tf.concat(batch_data_list, axis=1)
    
    #Create placeholders for every missing data type, with appropriate dimensions
    batch_data_list_observed = []
    for i in range(len(types_list)):
        batch_data_list_observed.append(tf.placeholder(tf.float32, shape=(batch_size,types_list[i]['dim'])))
    tf.concat(batch_data_list_observed, axis=1)
        
    #Create placeholders for the missing data indicator variable
    miss_list = tf.placeholder(tf.int32, shape=(batch_size,len(types_list)))
    
    #Placeholder for Gumbel-softmax parameter
    tau = tf.placeholder(tf.float32,shape=())
    
    return batch_data_list, batch_data_list_observed, miss_list, tau, types_list

def batch_normalization(batch_data_list, types_list, miss_list):
    
    normalized_data = []
    normalization_parameters = []
    
    for i,d in enumerate(batch_data_list):
        #Partition the data in missing data (0) and observed data n(1)
        missing_data, observed_data = tf.dynamic_partition(d, miss_list[:,i], num_partitions=2)
        condition_indices = tf.dynamic_partition(tf.range(tf.shape(d)[0]), miss_list[:,i], num_partitions=2)
        
        if types_list[i]['type'] == 'real':
            #We transform the data to a gaussian with mean 0 and std 1
            data_mean, data_var = tf.nn.moments(observed_data,0)
            data_var = tf.clip_by_value(data_var,1e-6,1e20) #Avoid zero values
            aux_X = tf.nn.batch_normalization(observed_data,data_mean,data_var,offset=0.0,scale=1.0,variance_epsilon=1e-6)
            
            normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data, aux_X]))
            normalization_parameters.append([data_mean, data_var])
            
        #When using log-normal
        elif types_list[i]['type'] == 'pos':
#           #We transform the log of the data to a gaussian with mean 0 and std 1
            observed_data_log = tf.log(1 + observed_data)
            data_mean_log, data_var_log = tf.nn.moments(observed_data_log,0)
            data_var_log = tf.clip_by_value(data_var_log,1e-6,1e20) #Avoid zero values
            aux_X = tf.nn.batch_normalization(observed_data_log,data_mean_log,data_var_log,offset=0.0,scale=1.0,variance_epsilon=1e-6)
            
            normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data, aux_X]))
            normalization_parameters.append([data_mean_log, data_var_log])
            
        elif types_list[i]['type'] == 'count':
            
            #Input log of the data
            aux_X = tf.log(observed_data)
            
            normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data, aux_X]))
            normalization_parameters.append([0.0, 1.0])
            
            
        else:
            #Don't normalize the categorical and ordinal variables
            normalized_data.append(d)
            normalization_parameters.append([0.0, 1.0]) #No normalization here
    
    return normalized_data, normalization_parameters

def s_proposal_multinomial(X, batch_size, s_dim, tau, reuse):
    
    #We propose a categorical distribution to create a GMM for the latent space z
    log_pi = tf.layers.dense(inputs=X, units=s_dim, activation=None,
                         kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_1_' + 'enc_s', reuse=reuse)
    
    #Gumbel-softmax trick
    U = -tf.log(-tf.log(tf.random_uniform([batch_size,s_dim])))
    samples_s = tf.nn.softmax((log_pi + U)/tau)
    
    return samples_s, log_pi

def z_proposal_GMM(X, samples_s, batch_size, z_dim, reuse):
    
    #We propose a GMM for z
    mean_qz = tf.layers.dense(inputs=tf.concat([X,samples_s],1), units=z_dim, activation=None,
                         kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_1_' + 'mean_enc_z', reuse=reuse)
    log_var_qz = tf.layers.dense(inputs=tf.concat([X,samples_s],1), units=z_dim, activation=None,
                         kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_1_' + 'logvar_enc_z', reuse=reuse)
    
    # Avoid numerical problems
    log_var_qz = tf.clip_by_value(log_var_qz,-15.0,15.0)
    # Rep-trick
    eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32)
    samples_z = mean_qz+tf.multiply(tf.exp(log_var_qz/2), eps)
    
    return samples_z, [mean_qz, log_var_qz]

def z_proposal_GMM_factorized(X, samples_s, miss_list, batch_size, z_dim, reuse):
    
    mean_qz = []
    log_var_qz = []
    
    for i,d in enumerate(X):
        
        #Partition the data in missing data (0) and observed data n(1)
        missing_data, observed_data = tf.dynamic_partition(d, miss_list[:,i], num_partitions=2)
        missing_s, observed_s = tf.dynamic_partition(samples_s, miss_list[:,i], num_partitions=2)
        condition_indices = tf.dynamic_partition(tf.range(tf.shape(d)[0]), miss_list[:,i], num_partitions=2)
        
        #Get the dimensions of the observed data
        nObs = tf.shape(observed_data)[0]
        
        #Mean layer
        aux_m = tf.layers.dense(inputs=tf.concat([observed_data,observed_s],1), units=z_dim, activation=None,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_1_' + 'mean_enc_z'+str(i), reuse=reuse)
        #Reconstruct means with zeros (so they don't affect the mean_joint)
        aux_mean_qz = tf.dynamic_stitch(condition_indices, [tf.zeros([batch_size-nObs,z_dim],dtype=tf.float32),aux_m])

        #Logvar layers
        aux_lv = tf.layers.dense(inputs=tf.concat([observed_data,observed_s],1), units=z_dim, activation=None,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_1_' + 'logvar_enc_z'+str(i), reuse=reuse)
        #Set a high value to make the variance in the missing cases negligible
        aux_log_var_qz = tf.dynamic_stitch(condition_indices, [tf.fill([batch_size-nObs,z_dim],15.0),aux_lv])
        
        mean_qz.append(aux_mean_qz)
        log_var_qz.append(aux_log_var_qz)
        
    #Input prior
    log_var_qz.append(tf.zeros([batch_size,z_dim]))
    mean_qz.append(tf.zeros([batch_size,z_dim]))
    #Compute full parameters, as a product of Gaussians distribution
    log_var_qz_joint = -tf.reduce_logsumexp(tf.negative(log_var_qz), 0)
    mean_qz_joint = tf.multiply(tf.exp(log_var_qz_joint), tf.reduce_sum(tf.multiply(mean_qz,tf.exp(tf.negative(log_var_qz))), 0))
    
    # Avoid numerical problems
    log_var_qz = tf.clip_by_value(log_var_qz,-15.0,15.0)
    # Rep-trick
    eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32)
    samples_z = mean_qz_joint+tf.multiply(tf.exp(log_var_qz_joint/2), eps)
    
    return samples_z, [mean_qz_joint, log_var_qz_joint]

def z_distribution_GMM(samples_s, z_dim, reuse):
    
    #We propose a GMM for z
    mean_pz = tf.layers.dense(inputs=samples_s, units=z_dim, activation=None,
                         kernel_initializer=tf.random_normal_initializer(stddev=0.05), name= 'layer_1_' + 'mean_dec_z', reuse=reuse)
    log_var_pz = tf.zeros([tf.shape(samples_s)[0],z_dim])
    
    # Avoid numerical problems
    log_var_pz = tf.clip_by_value(log_var_pz,-15.0,15.0)
    
    return mean_pz, log_var_pz

def y_partition(samples_y, types_list, y_dim_partition):
    
    grouped_samples_y = []
    #First element must be 0 and the length of the partition vector must be len(types_dict)+1
    if len(y_dim_partition) != len(types_list):
        raise Exception("The length of the partition vector must match the number of variables in the data + 1")
        
    #Insert a 0 at the beginning of the cumsum vector
    partition_vector_cumsum = np.insert(np.cumsum(y_dim_partition),0,0)
    for i in range(len(types_list)):
        grouped_samples_y.append(samples_y[:,partition_vector_cumsum[i]:partition_vector_cumsum[i+1]])
    
    return grouped_samples_y

def theta_estimation_from_y(samples_y, types_list, miss_list, batch_size, reuse):
    
    theta = []
    
    #Independet yd -> Compute p(xd|yd)
    for i,d in enumerate(samples_y):
        
        #Partition the data in missing data (0) and observed data (1)
        missing_y, observed_y = tf.dynamic_partition(d, miss_list[:,i], num_partitions=2)
        condition_indices = tf.dynamic_partition(tf.range(tf.shape(d)[0]), miss_list[:,i], num_partitions=2)
        nObs = tf.shape(observed_y)[0]
        
        #Different layer models for each type of variable
        if types_list[i]['type'] == 'real':
            params = theta_real(observed_y, missing_y, condition_indices, types_list, nObs, batch_size, i, reuse)
        
        elif types_list[i]['type'] == 'pos':
            params = theta_pos(observed_y, missing_y, condition_indices, types_list, nObs, batch_size, i, reuse)
            
        elif types_list[i]['type'] == 'count':
            params = theta_count(observed_y, missing_y, condition_indices, types_list, nObs, batch_size, i, reuse)
        
        elif types_list[i]['type'] == 'cat':
            params = theta_cat(observed_y, missing_y, condition_indices, types_list, nObs, batch_size, i, reuse)
            
        elif types_list[i]['type'] == 'ordinal':
            params = theta_ordinal(observed_y, missing_y, condition_indices, types_list, nObs, batch_size, i, reuse)
            
        theta.append(params)
            
    return theta

def theta_real(observed_y, missing_y, condition_indices, types_list, nObs, batch_size, i, reuse):
    
    #Mean layer
    h2_mean = observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), reuse=reuse)
    #Sigma Layer
    h2_sigma = observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2_sigma' + str(i), reuse=reuse)
    
    return [h2_mean, h2_sigma]

def theta_pos(observed_y, missing_y, condition_indices, types_list, nObs, batch_size, i, reuse):
    
    #Mean layer
    h2_mean = observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), reuse=reuse)
    #Sigma Layer
    h2_sigma = observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2_sigma' + str(i), reuse=reuse)
    
    return [h2_mean, h2_sigma]

def theta_count(observed_y, missing_y, condition_indices, types_list, nObs, batch_size, i, reuse):
    
    #Lambda Layer
    h2_lambda = observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), reuse=reuse)
    
    return h2_lambda

def theta_cat(observed_y, missing_y, condition_indices, types_list, nObs, batch_size, i, reuse):
    
    #Log pi layer, with zeros in the first value to avoid the identificability problem
    h2_log_pi_partial = observed_data_layer(observed_y, missing_y, condition_indices, output_dim=int(types_list[i]['dim'])-1, name='layer_h2' + str(i), reuse=reuse)
    h2_log_pi = tf.concat([tf.zeros([batch_size,1]), h2_log_pi_partial],1)
    
    return h2_log_pi

def theta_ordinal(observed_y, missing_y, condition_indices, types_list, nObs, batch_size, i, reuse):
    
    #Theta layer, Dimension of ordinal - 1
    h2_theta = observed_data_layer(observed_y, missing_y, condition_indices, output_dim=int(types_list[i]['dim'])-1, name='layer_h2' + str(i), reuse=reuse)
    #Mean layer, a single value
    h2_mean = observed_data_layer(observed_y, missing_y, condition_indices, output_dim=1, name='layer_h2_sigma' + str(i), reuse=reuse)
    
    return [h2_theta, h2_mean]

def observed_data_layer(observed_data, missing_data, condition_indices, output_dim, name, reuse):
    
    #Train a layer with the observed data and reuse it for the missing data
    obs_output = tf.layers.dense(inputs=observed_data, units=output_dim, activation=None,
                         kernel_initializer=tf.random_normal_initializer(stddev=0.05),name=name,reuse=reuse,trainable=True)
    miss_output = tf.layers.dense(inputs=missing_data, units=output_dim, activation=None,
                     kernel_initializer=tf.random_normal_initializer(stddev=0.05),name=name,reuse=True,trainable=False)
    #Join back the data
    output = tf.dynamic_stitch(condition_indices, [miss_output,obs_output])
    
    return output


def loglik_evaluation(batch_data_list, types_list, miss_list, theta, normalization_params, reuse):
    
    log_p_x = []
    log_p_x_missing = []
    samples_x = []
    params_x = []
    
    #Independet yd -> Compute log(p(xd|yd))
    for i,d in enumerate(batch_data_list):
        
        # Select the likelihood for the types of variables
        loglik_function = getattr(loglik_models_missing_normalize, 'loglik_' + types_list[i]['type'])
        
        out = loglik_function([d,miss_list[:,i]], types_list[i], theta[i], normalization_params[i],
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_1_mean_dec_x' + str(i), reuse=reuse)
            
        log_p_x.append(out['log_p_x'])
        log_p_x_missing.append(out['log_p_x_missing']) #Test-loglik element
        samples_x.append(out['samples'])
        params_x.append(out['params'])
        
    return log_p_x, log_p_x_missing, samples_x, params_x