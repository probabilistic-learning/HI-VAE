#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:40:14 2017

Functions needed to read the data from different databases

@author: anazabal, olmosUC3M, ivaleraM
"""

import csv
import numpy as np
import os
from sklearn.metrics import mean_squared_error

def read_data(data_file, types_file, miss_file, true_miss_file):
    
    #Read types of data from data file
    with open(types_file) as f:
        types_dict = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
    
    #Read data from input file
    with open(data_file, 'r') as f:
        data = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
        data = np.array(data)
    
    #Sustitute NaN values by something (we assume we have the real missing value mask)
    if true_miss_file:
        with open(true_miss_file, 'r') as f:
            missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            missing_positions = np.array(missing_positions)
            
        true_miss_mask = np.ones([np.shape(data)[0],len(types_dict)])
        true_miss_mask[missing_positions[:,0]-1,missing_positions[:,1]-1] = 0 #Indexes in the csv start at 1
        data_masked = np.ma.masked_where(np.isnan(data),data) 
        #We need to fill the data depending on the given data...
        data_filler = []
        for i in range(len(types_dict)):
            if types_dict[i]['type'] == 'cat' or types_dict[i]['type'] == 'ordinal':
                aux = np.unique(data[:,i])
                data_filler.append(aux[0])  #Fill with the first element of the cat (0, 1, or whatever)
            else:
                data_filler.append(0.0)
            
        data = data_masked.filled(data_filler)
    else:
        true_miss_mask = np.ones([np.shape(data)[0],len(types_dict)]) #It doesn't affect our data
    
    #Construct the data matrices
    data_complete = []
    for i in range(np.shape(data)[1]):
        
        if types_dict[i]['type'] == 'cat':
            #Get categories
            cat_data = [int(x) for x in data[:,i]]
            categories, indexes = np.unique(cat_data,return_inverse=True)
            #Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(types_dict[i]['dim']))
            cat_data = new_categories[indexes]
            #Create one hot encoding for the categories
            aux = np.zeros([np.shape(data)[0],len(new_categories)])
            aux[np.arange(np.shape(data)[0]),cat_data] = 1
            data_complete.append(aux)
            
        elif types_dict[i]['type'] == 'ordinal':
            #Get categories
            cat_data = [int(x) for x in data[:,i]]
            categories, indexes = np.unique(cat_data,return_inverse=True)
            #Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(types_dict[i]['dim']))
            cat_data = new_categories[indexes]
            #Create thermometer encoding for the categories
            aux = np.zeros([np.shape(data)[0],1+len(new_categories)])
            aux[:,0] = 1
            aux[np.arange(np.shape(data)[0]),1+cat_data] = -1
            aux = np.cumsum(aux,1)
            data_complete.append(aux[:,:-1])
            
        else:
            data_complete.append(np.transpose([data[:,i]]))
                    
    data = np.concatenate(data_complete,1)
    
        
    #Read Missing mask from csv (contains positions of missing values)
    n_samples = np.shape(data)[0]
    n_variables = len(types_dict)
    miss_mask = np.ones([np.shape(data)[0],n_variables])
    #If there is no mask, assume all data is observed
    if os.path.isfile(miss_file):
        with open(miss_file, 'r') as f:
            missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            missing_positions = np.array(missing_positions)
        miss_mask[missing_positions[:,0]-1,missing_positions[:,1]-1] = 0 #Indexes in the csv start at 1
        
    return data, types_dict, miss_mask, true_miss_mask, n_samples


def next_batch(data, types_dict, miss_mask, batch_size, index_batch):
    
    #Create minibath
    batch_xs = data[index_batch*batch_size:(index_batch+1)*batch_size, :]
    
    #Slipt variables of the batches
    data_list = []
    initial_index = 0
    for d in types_dict:
        dim = int(d['dim'])
        data_list.append(batch_xs[:,initial_index:initial_index+dim])
        initial_index += dim
    
    #Missing data
    miss_list = miss_mask[index_batch*batch_size:(index_batch+1)*batch_size, :]
    
    return data_list, miss_list

def samples_concatenation(samples):
    
    for i,batch in enumerate(samples):
        if i == 0:
            samples_x = np.concatenate(batch['x'],1)
            samples_y = batch['y']
            samples_z = batch['z']
            samples_s = batch['s']
        else:
            samples_x = np.concatenate([samples_x,np.concatenate(batch['x'],1)],0)
            samples_y = np.concatenate([samples_y,batch['y']],0)
            samples_z = np.concatenate([samples_z,batch['z']],0)
            samples_s = np.concatenate([samples_s,batch['s']],0)
        
    return samples_s, samples_z, samples_y, samples_x

def discrete_variables_transformation(data, types_dict):
    
    ind_ini = 0
    output = []
    for d in range(len(types_dict)):
        ind_end = ind_ini + int(types_dict[d]['dim'])
        if types_dict[d]['type'] == 'cat':
            output.append(np.reshape(np.argmax(data[:,ind_ini:ind_end],1),[-1,1]))
        elif types_dict[d]['type'] == 'ordinal':
            output.append(np.reshape(np.sum(data[:,ind_ini:ind_end],1) - 1,[-1,1]))
        else:
            output.append(data[:,ind_ini:ind_end])
        ind_ini = ind_end
    
    return np.concatenate(output,1)

#Several baselines
def mean_imputation(train_data, miss_mask, types_dict):
    
    ind_ini = 0
    est_data = []
    for dd in range(len(types_dict)):
        #Imputation for cat and ordinal is done using the mode of the data
        if types_dict[dd]['type']=='cat' or types_dict[dd]['type']=='ordinal':
            ind_end = ind_ini + 1
            #The imputation is based on whatever is observed
            miss_pattern = (miss_mask[:,dd]==1)
            values, counts = np.unique(train_data[miss_pattern,ind_ini:ind_end],return_counts=True)
            data_mode = np.argmax(counts)
            data_imputed = train_data[:,ind_ini:ind_end]*miss_mask[:,ind_ini:ind_end] + data_mode*(1.0-miss_mask[:,ind_ini:ind_end])
            
        #Imputation for the rest of the variables is done with the mean of the data
        else:
            ind_end = ind_ini + int(types_dict[dd]['dim'])
            miss_pattern = (miss_mask[:,dd]==1)
            #The imputation is based on whatever is observed
            data_mean = np.mean(train_data[miss_pattern,ind_ini:ind_end],0)
            data_imputed = train_data[:,ind_ini:ind_end]*miss_mask[:,ind_ini:ind_end] + data_mean*(1.0-miss_mask[:,ind_ini:ind_end])
            
        est_data.append(data_imputed)
        ind_ini = ind_end
    
    return np.concatenate(est_data,1)

def p_distribution_params_concatenation(params,types_dict,z_dim,s_dim):
    
    keys = params[0].keys()
    out_dict = {key: [] for key in keys}
    
    for i,batch in enumerate(params):
        
        for d,k in enumerate(keys):
            
            if k == 'z' or k == 'y':
                if i == 0:
                    out_dict[k] = batch[k]
                else:
                    out_dict[k] = np.concatenate([out_dict[k],batch[k]],1)
                    
            elif k == 'x':
                if i == 0:
                    out_dict[k] = batch[k]
                else:
                    for v in range(len(types_dict)):
                        if types_dict[v]['type'] == 'pos' or types_dict[v]['type'] == 'real':
                            out_dict[k][v] = np.concatenate([out_dict[k][v],batch[k][v]],1)
                        else:
                            out_dict[k][v] = np.concatenate([out_dict[k][v],batch[k][v]],0)
        
    return out_dict

def q_distribution_params_concatenation(params,z_dim,s_dim):
    
    keys = params[0].keys()
    out_dict = {key: [] for key in keys}
    
    for i,batch in enumerate(params):
        for d,k in enumerate(keys):
            out_dict[k].append(batch[k])
            
    out_dict['z'] = np.concatenate(out_dict['z'],1)
    out_dict['s'] = np.concatenate(out_dict['s'],0)
        
    return out_dict

def statistics(loglik_params,types_dict):
    
    loglik_mean = []
    loglik_mode = []
    
    for d,attrib in enumerate(loglik_params):
        if types_dict[d]['type'] == 'real':
            #Normal distribution (mean, sigma)
            loglik_mean.append(attrib[0])
            loglik_mode.append(attrib[0])
        #Only for log-normal
        elif types_dict[d]['type'] == 'pos':
            #Log-normal distribution (mean, sigma)
            loglik_mean.append(np.exp(attrib[0] + 0.5*attrib[1]) - 1.0)
            loglik_mode.append(np.exp(attrib[0] - attrib[1]) - 1.0)
        elif types_dict[d]['type'] == 'count':
            #Poisson distribution (lambda)
            loglik_mean.append(attrib)
            loglik_mode.append(np.floor(attrib))
        
        else:
            #Categorical and ordinal (mode imputation for both)
            loglik_mean.append(np.reshape(np.argmax(attrib,1),[-1,1]))
            loglik_mode.append(np.reshape(np.argmax(attrib,1),[-1,1]))
        
            
    return np.transpose(np.squeeze(loglik_mean)), np.transpose(np.squeeze(loglik_mode))

def error_computation(x_train, x_hat, types_dict, miss_mask):
    
    error_observed = []
    error_missing = []
    ind_ini = 0
    for dd in range(len(types_dict)):
        #Mean classification error
        if types_dict[dd]['type']=='cat':
            ind_end = ind_ini + 1
            error_observed.append(np.mean(x_train[miss_mask[:,dd]==1,ind_ini:ind_end] != x_hat[miss_mask[:,dd]==1,ind_ini:ind_end]))
            if np.sum(miss_mask[:,dd]==0,0) == 0:
                error_missing.append(0)
            else:
                error_missing.append(np.mean(x_train[miss_mask[:,dd]==0,ind_ini:ind_end] != x_hat[miss_mask[:,dd]==0,ind_ini:ind_end]))
        #Mean "shift" error        
        elif types_dict[dd]['type']=='ordinal':
            ind_end = ind_ini + 1
            error_observed.append(np.mean(np.abs(x_train[miss_mask[:,dd]==1,ind_ini:ind_end] -x_hat[miss_mask[:,dd]==1,ind_ini:ind_end]))/int(types_dict[dd]['dim']))
            if np.sum(miss_mask[:,dd]==0,0) == 0:
                error_missing.append(0)
            else:
                error_missing.append(np.mean(np.abs(x_train[miss_mask[:,dd]==0,ind_ini:ind_end] -x_hat[miss_mask[:,dd]==0,ind_ini:ind_end]))/int(types_dict[dd]['dim']))
        #Normalized root mean square error
        else:
            ind_end = ind_ini + int(types_dict[dd]['dim'])
            norm_term = np.max(x_train[miss_mask[:,dd]==1,dd]) - np.min(x_train[miss_mask[:,dd]==1,dd])
            error_observed.append(np.sqrt(mean_squared_error(x_train[miss_mask[:,dd]==1,ind_ini:ind_end],x_hat[miss_mask[:,dd]==1,ind_ini:ind_end]))/norm_term)
            if np.sum(miss_mask[:,dd]==0,0) == 0:
                error_missing.append(0)
            else:
                error_missing.append(np.sqrt(mean_squared_error(x_train[miss_mask[:,dd]==0,ind_ini:ind_end],x_hat[miss_mask[:,dd]==0,ind_ini:ind_end]))/norm_term)
                
        ind_ini = ind_end
                
    return error_observed, error_missing