#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:49:42 2018

@author: anazabal, olmosUC3M, ivaleraM
"""

import sys
import argparse
import tensorflow as tf
import graph_new
import parser_arguments
import time
import numpy as np
#import plot_functions
import read_functions
import os
import csv



def print_loss(epoch, start_time, avg_loss, avg_test_loglik, avg_KL_s, avg_KL_z):
    print("Epoch: [%2d]  time: %4.4f, train_loglik: %.8f, KL_z: %.8f, KL_s: %.8f, ELBO: %.8f, Test_loglik: %.8f"
          % (epoch, time.time() - start_time, avg_loss, avg_KL_z, avg_KL_s, avg_loss-avg_KL_z-avg_KL_s, avg_test_loglik))

#Get arguments for parser
args = parser_arguments.getArgs(sys.argv[1:])

#Create a directoy for the save file
if not os.path.exists('./Saved_Networks/' + args.save_file):
    os.makedirs('./Saved_Networks/' + args.save_file)

network_file_name='./Saved_Networks/' + args.save_file + '/' + args.save_file +'.ckpt'
log_file_name='./Saved_Network/' + args.save_file + '/log_file_' + args.save_file +'.txt'

print(args)

train_data, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
#Check batch size
if args.batch_size > n_samples:
    args.batch_size = n_samples
#Get an integer number of batches
n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))
#Compute the real miss_mask
miss_mask = np.multiply(miss_mask, true_miss_mask)

#Creating graph
sess_HVAE = tf.Graph()

with sess_HVAE.as_default():
    
    tf_nodes = graph_new.HVAE_graph(args.model_name, args.types_file, args.batch_size,
                                learning_rate=1e-3, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, y_dim_partition=args.dim_latent_y_partition)

################### Running the VAE Training #################################



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
        error_train_mode_global = []
        error_test_mode_global = []
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
            tau = np.max([1.0 - 0.01*epoch,1e-3])
#            tau = 1e-3
            tau2 = np.min([0.001*epoch,1.0])
            
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
                
##                Delete not known data (mean inputation)
#                data_list_observed = []
##                data_mean = np.mean(train_data_aux,0)
##                initial_index = 0
#                for d in range(len(data_list)):
#                    data_mean = np.mean(data_list[d][miss_list[:,d]==1,:],0)
##                    dim = int(types_dict[i]['dim'])
#                    if types_dict[d]['type'] == 'real' or types_dict[d]['type'] == 'pos':
#                        data_mean = np.mean(data_list[d][miss_list[:,d]==1,:],0)
#                        data_list_observed.append(data_list[d]*np.reshape(miss_list[:,d],[args.batch_size,1]) + data_mean*np.reshape(1-miss_list[:,d],[args.batch_size,1]))
#                    elif types_dict[d]['type'] == 'cat':
#                        data_median = (np.mean(data_list[d][miss_list[:,d]==1,:],0)==np.max(np.mean(data_list[d][miss_list[:,d]==1,:],0))).astype(float)
#                        data_list_observed.append(data_list[d]*np.reshape(miss_list[:,d],[args.batch_size,1]) + data_median*np.reshape(1-miss_list[:,d],[args.batch_size,1]))
#                    elif types_dict[d]['type'] == 'ordinal':
#                        data_median = (np.mean(data_list[d][miss_list[:,d]==1,:],0)>=0.5).astype(float)
#                        data_list_observed.append(data_list[d]*np.reshape(miss_list[:,d],[args.batch_size,1]) + data_median*np.reshape(1-miss_list[:,d],[args.batch_size,1]))
#                    else:
#                        data_median = np.median(data_list[d][miss_list[:,d]==1,:],0)
#                        data_list_observed.append(data_list[d]*np.reshape(miss_list[:,d],[args.batch_size,1]) + data_median*np.reshape(1-miss_list[:,d],[args.batch_size,1]))
##                        data_list_observed.append(data_list[d]*np.reshape(miss_list[:,d],[args.batch_size,1]))
##                    data_list_observed.append(data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) + data_mean*np.reshape(1-miss_list[:,i],[args.batch_size,1]))
##                    initial_index += dim
                
                #Create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list
                feedDict[tf_nodes['tau_GS']] = tau
                feedDict[tf_nodes['tau_var']] = tau2
                
                #Running VAE
                _,loss,KL_z,KL_s,samples,log_p_x,log_p_x_missing,p_params,q_params  = session.run([tf_nodes['optim'], tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'], tf_nodes['samples'],
                                                         tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],tf_nodes['p_params'],tf_nodes['q_params']],
                                                         feed_dict=feedDict)
                
                samples_test,log_p_x_test,log_p_x_missing_test,test_params  = session.run([tf_nodes['samples_test'],tf_nodes['log_p_x_test'],tf_nodes['log_p_x_missing_test'],tf_nodes['test_params']],
                                                             feed_dict=feedDict)
                
#                #Collect all samples, distirbution parameters and logliks in lists
#                samples_list.append(samples)
#                p_params_list.append(p_params)
#                q_params_list.append(q_params)
#                log_p_x_total.append(log_p_x)
#                log_p_x_missing_total.append(log_p_x_missing)
                
                #Evaluate results on the imputation with mode, not on the samlpes!
                samples_list.append(samples_test)
                p_params_list.append(test_params)
    #                        p_params_list.append(p_params)
                q_params_list.append(q_params)
                log_p_x_total.append(log_p_x_test)
                log_p_x_missing_total.append(log_p_x_missing_test)
                
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
            
#            est_data_transformed[np.isinf(est_data_transformed)] = 1e20
            
            #Create global dictionary of the distribution parameters
            p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, types_dict, args.dim_latent_z, args.dim_latent_s)
            q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list,  args.dim_latent_z, args.dim_latent_s)
            
            #Number of clusters created
            cluster_index = np.argmax(q_params_complete['s'],1)
            cluster = np.unique(cluster_index)
            print('Clusters: ' + str(len(cluster)))
            
            #Compute mean and mode of our loglik models
            loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'],types_dict)
#            loglik_mean[np.isinf(loglik_mean)] = 1e20
                
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
                print('Test error mode: ' + str(np.round(np.mean(error_test_mode),3)))
                print("")
                
            #Compute train and test loglik per variables
            loglik_per_variable = np.sum(log_p_x_total,0)/np.sum(miss_mask_aux,0)
            loglik_per_variable_missing = np.sum(log_p_x_missing_total,0)/np.sum(1.0-miss_mask_aux,0)
            
            #Store evolution of all the terms in the ELBO
            loglik_epoch.append(loglik_per_variable)
            testloglik_epoch.append(loglik_per_variable_missing)
            KL_s_epoch.append(avg_KL_s/n_batches)
            KL_z_epoch.append(avg_KL_z/n_batches)
            error_train_mode_global.append(error_train_mode)
            error_test_mode_global.append(error_test_mode)
            
            
            if epoch % args.save == 0:
                print('Saving Variables ...')  
                save_path = saver.save(session, network_file_name)
                
        
            
        print('Training Finished ...')
        
        #Saving needed variables in csv
        if not os.path.exists('./Results_csv/' + args.save_file):
            os.makedirs('./Results_csv/' + args.save_file)
        
        with open('Results_csv/' + args.save_file + '/' + args.save_file + '_loglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(loglik_epoch)
            
        with open('Results_csv/' + args.save_file + '/' + args.save_file + '_testloglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(testloglik_epoch)
            
        with open('Results_csv/' + args.save_file + '/' + args.save_file + '_KL_s.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(np.reshape(KL_s_epoch,[-1,1]))
            
        with open('Results_csv/' + args.save_file + '/' + args.save_file + '_KL_z.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(np.reshape(KL_z_epoch,[-1,1]))
            
        with open('Results_csv/' + args.save_file + '/' + args.save_file + '_train_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(error_train_mode_global)
            
        with open('Results_csv/' + args.save_file + '/' + args.save_file + '_test_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(error_test_mode_global)
            
        # Save the variables to disk at the end
        save_path = saver.save(session, network_file_name) 
        
        
    #Test phase
    else:
        
        start_time = time.time()
        # Training cycle
        
#        f_toy2, ax_toy2 = plt.subplots(4,4,figsize=(8, 8))
        loglik_epoch = []
        testloglik_epoch = []
        error_train_mode_global = []
        error_test_mode_global = []
        error_imputed_global = []
        est_data_transformed_total = []
        
        #Only one epoch needed, since we are doing mode imputation
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
            
            label_ind = 2
            
            # Constant Gumbel-Softmax parameter (where we have finished the annealing)
            tau = 1e-3
#            tau = 1.0
            
            #Randomize the data in the mini-batches
    #        random_perm = np.random.permutation(range(np.shape(data)[0]))
            random_perm = range(np.shape(train_data)[0])
            train_data_aux = train_data[random_perm,:]
            miss_mask_aux = miss_mask[random_perm,:]
            true_miss_mask_aux = true_miss_mask[random_perm,:]
            
            for i in range(n_batches):      
                
                #Create train minibatch
                data_list, miss_list = read_functions.next_batch(train_data_aux, types_dict, miss_mask_aux, args.batch_size, 
                                                                 index_batch=i)
#                print(np.mean(data_list[0],0))
    
                #Delete not known data
                data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) for i in range(len(data_list))]
              
                
                #Create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list
                feedDict[tf_nodes['tau_GS']] = tau
                
                #Get samples from the model
                loss,KL_z,KL_s,samples,log_p_x,log_p_x_missing,p_params,q_params  = session.run([tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'], tf_nodes['samples'],
                                                             tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],tf_nodes['p_params'],tf_nodes['q_params']],
                                                             feed_dict=feedDict)
                
                samples_test,log_p_x_test,log_p_x_missing_test,test_params  = session.run([tf_nodes['samples_test'],tf_nodes['log_p_x_test'],tf_nodes['log_p_x_missing_test'],tf_nodes['test_params']],
                                                             feed_dict=feedDict)
                
                
                samples_list.append(samples_test)
                p_params_list.append(test_params)
    #                        p_params_list.append(p_params)
                q_params_list.append(q_params)
                log_p_x_total.append(log_p_x_test)
                log_p_x_missing_total.append(log_p_x_missing_test)
                
    #            # Compute average loss
    #            avg_loss += np.mean(loss)
    #            avg_KL_s += np.mean(KL_s)
    #            avg_KL_z += np.mean(KL_z)
                
            #Separate the samples from the batch list
            s_aux, z_aux, y_total, est_data = read_functions.samples_concatenation(samples_list)
            
            #Transform discrete variables to original values
            train_data_transformed = read_functions.discrete_variables_transformation(train_data_aux[:n_batches*args.batch_size,:], types_dict)
            est_data_transformed = read_functions.discrete_variables_transformation(est_data, types_dict)
            est_data_imputed = read_functions.mean_imputation(train_data_transformed, miss_mask_aux[:n_batches*args.batch_size,:], types_dict)
            
            #Create global dictionary of the distribution parameters
            p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, types_dict, args.dim_latent_z, args.dim_latent_s)
            q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list,  args.dim_latent_z, args.dim_latent_s)
            
            #Number of clusters created
            cluster_index = np.argmax(q_params_complete['s'],1)
            cluster = np.unique(cluster_index)
            print('Clusters: ' + str(len(cluster)))
            
            #Compute mean and mode of our loglik models
            loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'],types_dict)
    
            #Try this for the errors
            error_train_mean, error_test_mean = read_functions.error_computation(train_data_transformed, loglik_mean, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
            error_train_mode, error_test_mode = read_functions.error_computation(train_data_transformed, loglik_mode, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
            error_train_samples, error_test_samples = read_functions.error_computation(train_data_transformed, est_data_transformed, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
            error_train_imputed, error_test_imputed = read_functions.error_computation(train_data_transformed, est_data_imputed, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
                
    #            Compute test-loglik from log_p_x_missing
            log_p_x_missing_total = np.transpose(np.concatenate(log_p_x_missing_total,1))
            if args.true_miss_file:
                log_p_x_missing_total = np.multiply(log_p_x_missing_total,true_miss_mask_aux[:n_batches*args.batch_size,:])
            avg_test_loglik = np.sum(log_p_x_missing_total)/np.sum(1.0-miss_mask_aux)
    
            # Display logs per epoch step
            if args.display == 1:
    #            print_loss(0, start_time, avg_loss/n_batches, avg_test_loglik, avg_KL_s/n_batches, avg_KL_z/n_batches)
                print(np.round(error_test_mode,3))
                print('Test error mode: ' + str(np.round(np.mean(error_test_mode),3)))
                print("")
                
            
                
            #Plot evolution of test loglik
            loglik_per_variable = np.sum(np.concatenate(log_p_x_total,1),1)/np.sum(miss_mask,0)
            loglik_per_variable_missing = np.sum(log_p_x_missing_total,0)/np.sum(1.0-miss_mask,0)
            
            loglik_epoch.append(loglik_per_variable)
            testloglik_epoch.append(loglik_per_variable_missing)
            
            print('Test loglik: ' + str(np.round(np.mean(loglik_per_variable_missing),3)))
            
            
            #Re-run test error mode
            error_train_mode_global.append(error_train_mode)
            error_test_mode_global.append(error_test_mode)
            error_imputed_global.append(error_test_imputed)
            
            #Store data samples
            est_data_transformed_total.append(est_data_transformed)
            
        #Compute the data reconstruction
        data_reconstruction = train_data_transformed * miss_mask_aux[:n_batches*args.batch_size,:] + \
                                np.round(loglik_mode,3) * (1 - miss_mask_aux[:n_batches*args.batch_size,:])
        
        
#        data_reconstruction = -1 * miss_mask_aux[:n_batches*args.batch_size,:] + \
#                                np.round(loglik_mode,3) * (1 - miss_mask_aux[:n_batches*args.batch_size,:])
                                
        train_data_transformed = train_data_transformed[np.argsort(random_perm)]
        data_reconstruction = data_reconstruction[np.argsort(random_perm)]
        
        if not os.path.exists('./Results/' + args.save_file):
            os.makedirs('./Results/' + args.save_file)
            
        with open('Results/' + args.save_file + '/' + args.save_file + '_data_reconstruction.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(data_reconstruction)
        with open('Results/' + args.save_file + '/' + args.save_file + '_data_true.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train_data_transformed)
            
            
        #Saving needed variables in csv
        if not os.path.exists('./Results_test_csv/' + args.save_file):
            os.makedirs('./Results_test_csv/' + args.save_file)
        
        #Train loglik per variable
        with open('Results_test_csv/' + args.save_file + '/' + args.save_file + '_loglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(loglik_epoch)
            
        #Test loglik per variable
        with open('Results_test_csv/' + args.save_file + '/' + args.save_file + '_testloglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(testloglik_epoch)
            
        #Train NRMSE per variable
        with open('Results_test_csv/' + args.save_file + '/' + args.save_file + '_train_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(error_train_mode_global)
            
        #Test NRMSE per variable
        with open('Results_test_csv/' + args.save_file + '/' + args.save_file + '_test_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(error_test_mode_global)
            
        #Number of clusters
        with open('Results_test_csv/' + args.save_file + '/' + args.save_file + '_clusters.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows([[len(cluster)]])
 
