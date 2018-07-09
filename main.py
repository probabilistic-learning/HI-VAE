#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:49:42 2018

@author: anazabal, olmosUC3M, ivaleraM
"""

import tensorflow as tf
import graph_new
import parser_arguments
import time
import numpy as np
from matplotlib import pyplot as plt
#import plot_functions
import read_functions
import os



def print_loss(epoch, start_time, avg_loss, avg_test_loglik, avg_KL_s, avg_KL_z):
    print("Epoch: [%2d]  time: %4.4f, train_loglik: %.8f, KL_z: %.8f, KL_s: %.8f, ELBO: %.8f, Test_loglik: %.8f"
          % (epoch, time.time() - start_time, avg_loss, avg_KL_z, avg_KL_s, avg_loss-avg_KL_z-avg_KL_s, avg_test_loglik))


plt.close('all')

##Parser for main settingsc
#parser = argparse.ArgumentParser()
#parser.add_argument('--settings')
#args_main = parser.parse_args()
#
## settings
#settings = args_main.settings
settings = '--epochs 100 --model_name model_HIVAE_inputDropout --restore 0 --train 1 \
            --data_file defaultCredit/data.csv --types_file defaultCredit/data_types.csv --miss_file defaultCredit/Missing30_1.csv \
            --batch_size 1000 --save 1001 --save_file model_test\
            --dim_latent_s 10 --dim_latent_z 10 --dim_latent_y 5 \
            --miss_percentage_train 0.2 --miss_percentage_test 0.5'
#            --true_miss_file Adult/MissingTrue.csv'
#            --dim_latent_y_partition 4 4 4 4 4 4 4 4 4 4 4 10 11
            
argvals = settings.split()
args = parser_arguments.getArgs(argvals)

#Create a directoy for the save file
if not os.path.exists('./Saved_Networks/' + args.save_file):
    os.makedirs('./Saved_Networks/' + args.save_file)

network_file_name='./Saved_Networks/' + args.save_file + '/' + args.save_file +'.ckpt'
log_file_name='./Saved_Network/' + args.save_file + '/log_file_' + args.save_file +'.txt'

print(args)

#Creating graph
sess_HVAE = tf.Graph()

with sess_HVAE.as_default():
    
    tf_nodes = graph_new.HVAE_graph(args.model_name, args.types_file, args.batch_size,
                                learning_rate=1e-3, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, y_dim_partition=args.dim_latent_y_partition)

################### Running the VAE Training #################################

train_data, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
#Get an integer number of batches
n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))
#Compute the real miss_mask
miss_mask = np.multiply(miss_mask, true_miss_mask)



with tf.Session(graph=sess_HVAE) as session:
        
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
        
    if(args.restore == 1):
        saver.restore(session, network_file_name)
        print("Model restored.")
    else:
#        saver = tf.train.Saver()
        print('Initizalizing Variables ...')
        tf.global_variables_initializer().run()
    
    print('Training the HVAE ...')
    if(args.train == 1):
        
        start_time = time.time()
        # Training cycle
        
        loglik_epoch = []
        testloglik_epoch = []
        KL_s_epoch = []
        KL_z_epoch = []
        for epoch in range(args.epochs):
            avg_loss = 0.
            avg_KL_s = 0.
            avg_KL_z = 0.
            samples_list = []
            p_params_list = []
            q_params_list = []
            log_p_x_total = []
            log_p_x_missing_total = []
            
            # Annealing of Gumbel-Softmax parameter
            tau = np.max([1.0 - 0.001*epoch,1e-3])
            
            #Randomize the data in the mini-batches
            random_perm = np.random.permutation(range(np.shape(train_data)[0]))
            train_data_aux = train_data[random_perm,:]
            miss_mask_aux = miss_mask[random_perm,:]
            true_miss_mask_aux = true_miss_mask[random_perm,:]
            
            for i in range(n_batches):      
                
                #Create inputs for the feed_dict
                data_list, miss_list = read_functions.next_batch(train_data_aux, types_dict, miss_mask_aux, args.batch_size, index_batch=i)

                #Delete not known data (input zeros)
                data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) for i in range(len(data_list))]
                
                #Create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list
                feedDict[tf_nodes['tau_GS']] = tau
                
                #Running VAE
                _,loss,KL_z,KL_s,samples,log_p_x,log_p_x_missing,p_params,q_params  = session.run([tf_nodes['optim'], tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'], tf_nodes['samples'],
                                                         tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],tf_nodes['p_params'],tf_nodes['q_params']],
                                                         feed_dict=feedDict)
                
                #Collect all samples, distirbution parameters and logliks in lists
                samples_list.append(samples)
                p_params_list.append(p_params)
                q_params_list.append(q_params)
                log_p_x_total.append(log_p_x)
                log_p_x_missing_total.append(log_p_x_missing)
                
                # Compute average loss
                avg_loss += np.mean(loss)
                avg_KL_s += np.mean(KL_s)
                avg_KL_z += np.mean(KL_z)
                
            #Concatenate samples in arrays
            s_total, z_total, y_total, est_data = read_functions.samples_concatenation(samples_list)
            
            #Transform discrete variables back to the original values
            train_data_transformed = read_functions.discrete_variables_transformation(train_data_aux[:n_batches*args.batch_size,:], types_dict)
            est_data_transformed = read_functions.discrete_variables_transformation(est_data, types_dict)
            est_data_imputed = read_functions.mean_imputation(train_data_transformed, miss_mask_aux[:n_batches*args.batch_size,:], types_dict)
            
            #Create global dictionary of the distribution parameters
            p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, types_dict, args.dim_latent_z, args.dim_latent_s)
            q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list,  args.dim_latent_z, args.dim_latent_s)
            
            #Compute mean and mode of our loglik models
            loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'],types_dict)
                
            #Try this for the errors
            error_train_mean, error_test_mean = read_functions.error_computation(train_data_transformed, loglik_mean, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
            error_train_mode, error_test_mode = read_functions.error_computation(train_data_transformed, loglik_mode, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
            error_train_samples, error_test_samples = read_functions.error_computation(train_data_transformed, est_data_transformed, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
            error_train_imputed, error_test_imputed = read_functions.error_computation(train_data_transformed, est_data_imputed, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
                
            #Compute test-loglik from log_p_x_missing
            log_p_x_total = np.transpose(np.concatenate(log_p_x_total,1))
            log_p_x_missing_total = np.transpose(np.concatenate(log_p_x_missing_total,1))
            if args.true_miss_file:
                log_p_x_missing_total = np.multiply(log_p_x_missing_total,true_miss_mask_aux[:n_batches*args.batch_size,:])
            avg_test_loglik = np.sum(log_p_x_missing_total)/np.sum(1.0-miss_mask_aux)

            # Display logs per epoch step
            if epoch % args.display == 0:
                print_loss(epoch, start_time, avg_loss/n_batches, avg_test_loglik, avg_KL_s/n_batches, avg_KL_z/n_batches)
                print("")
                
            #Compute train and test loglik per variables
            loglik_per_variable = np.sum(log_p_x_total,0)/np.sum(miss_mask_aux,0)
            loglik_per_variable_missing = np.sum(log_p_x_missing_total,0)/np.sum(1.0-miss_mask_aux,0)
            
            #Store evolution of all the terms in the ELBO
            loglik_epoch.append(loglik_per_variable)
            testloglik_epoch.append(loglik_per_variable_missing)
            KL_s_epoch.append(avg_KL_s/n_batches)
            KL_z_epoch.append(avg_KL_z/n_batches)
            
            
            if epoch % args.save == 0:
                print('Saving Variables ...')  
                save_path = saver.save(session, network_file_name)    
            
        print('Training Finished ...')
        
        
    #Test phase
    else:
        
        start_time = time.time()
        # Training cycle
        
        f_toy2, ax_toy2 = plt.subplots(4,4,figsize=(8, 8))
        loglik_epoch = []
        testloglik_epoch = []
        for epoch in range(args.epochs):
            avg_loss = 0.
            avg_KL_s = 0.
            avg_KL_y = 0.
            avg_KL_z = 0.
            samples_list = []
            p_params_list = []
            q_params_list = []
            log_p_x_total = []
            log_p_x_missing_total = []
            
            # Constant Gumbel-Softmax parameter (where we have finished the annealing)
            tau = 1e-3
            
            for i in range(n_batches):      
                
                #Create train minibatch
                data_list, miss_list = read_functions.next_batch(train_data, types_dict, miss_mask, args.batch_size, 
                                                                 index_batch=i, miss_percentage=args.miss_percentage_train)

                #Delete not known data
                data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) for i in range(len(data_list))]
                
                #Create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list
                feedDict[tf_nodes['tau_GS']] = tau
                
                #Get samples from the model
                KL_s,loss,samples,dec_data,log_p_x,log_p_x_missing,loss_total,KL_z,p_params,q_params  = session.run([tf_nodes['KL_s'], tf_nodes['loss_re'],tf_nodes['samples'],
                                                         tf_nodes['dec_data'],tf_nodes['log_p_x'],
                                                         tf_nodes['log_p_x_missing'],tf_nodes['loss'],
                                                         tf_nodes['KL_z'],tf_nodes['p_params'],tf_nodes['q_params']],
                                                         feed_dict=feedDict)
                
                #Get samples from the generator function (computing the mode of all distributions)
                samples_test,log_p_x_test,log_p_x_missing_test,test_params  = session.run([tf_nodes['samples_test'],tf_nodes['log_p_x_test'],tf_nodes['log_p_x_missing_test'],tf_nodes['test_params']],
                                                         feed_dict=feedDict)
                
                
                samples_list.append(samples)
                p_params_list.append(test_params)
                q_params_list.append(q_params)
                log_p_x_total.append(log_p_x)
                log_p_x_missing_total.append(log_p_x_missing)
                
                # Compute average loss
                avg_loss += np.mean(loss)
                avg_KL_s += np.mean(KL_s)
                avg_KL_z += np.mean(KL_z)
                
            #Separate the samples from the batch list
            s_aux, z_aux, y_total, est_data = read_functions.samples_concatenation(samples_list, args.dim_latent_z, args.dim_latent_s)
            
            #Transform discrete variables to original values
            train_data_transformed = read_functions.discrete_variables_transformation(train_data, types_dict)
            est_data_transformed = read_functions.discrete_variables_transformation(est_data, types_dict)
            est_data_imputed = read_functions.mean_imputation(train_data_transformed, miss_mask, types_dict)
            
            #Create global dictionary of the distribution parameters
            p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, types_dict, args.dim_latent_z, args.dim_latent_s)
            q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list,  args.dim_latent_z, args.dim_latent_s)
            
            #Compute mean and mode of our loglik models
            loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'],types_dict)
                
            #Try this for the errors
            error_train_mean, error_test_mean = read_functions.error_computation(train_data_transformed, loglik_mean, types_dict, miss_mask)
            error_train_mode, error_test_mode = read_functions.error_computation(train_data_transformed, loglik_mode, types_dict, miss_mask)
            error_train_samples, error_test_samples = read_functions.error_computation(train_data_transformed, est_data_transformed, types_dict, miss_mask)
            error_train_imputed, error_test_imputed = read_functions.error_computation(train_data_transformed, est_data_imputed, types_dict, miss_mask)
#            error_train_mean, error_test_mean_den = read_functions.error_computation_mean_den(train_data_transformed, loglik_mean, types_dict, miss_mask)
                
#            Compute test-loglik from log_p_x_missing
            log_p_x_missing_total = np.transpose(np.concatenate(log_p_x_missing_total,1))
            if args.true_miss_file:
                log_p_x_missing_total = np.multiply(log_p_x_missing_total,true_miss_mask)
            avg_test_loglik = np.sum(log_p_x_missing_total)/np.sum(1.0-miss_mask)

            # Display logs per epoch step
            if epoch % args.display == 0:
                print_loss(epoch, start_time, avg_loss/n_batches, avg_test_loglik, avg_KL_s/n_batches, avg_KL_z/n_batches)
                print("")
                
            #Plot evolution of test loglik
            loglik_per_variable = np.sum(np.concatenate(log_p_x_total,1),1)/np.sum(miss_mask,0)
            loglik_per_variable_missing = np.sum(log_p_x_missing_total,0)/np.sum(1.0-miss_mask,0)
            
            loglik_epoch.append(loglik_per_variable)
            testloglik_epoch.append(loglik_per_variable_missing)
 
