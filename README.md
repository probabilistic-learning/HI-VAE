# HI-VAE

This repository contains the implementation of our Heterogeneous Incomplete Variational Autoendoder model (HI-VAE). It has been written in Python, using Tensorflow.

The details of this model are included in this [paper](https://arxiv.org/abs/1807.03653). Please cite it if you use this code for your own research.

## Databse description

There are three different datasets considered in the experiments (Wine, Adult and Default Credit). Each dataset has each own folder, containing:

* **data.csv**: the dataset
* **data_types.csv**: a csv containing the types of that particular dataset. Every line is a different attribute containing three paramenters:
	* type: real, pos (positive), cat (categorical), ord (ordinal), count
	* dim: dimension of the variable
	* nclass: number of categories (for cat and ord)
* **Missingxx_y.csv**: a csv containing the positions of the different missing values in the data. Each "y" mask was generated randomly, containing a "xx" % of missing values.

You can add your own datasets as long as they follow this structure.

## Files description

* **main.py**: In the variable settings you can select the different parameters for our model. A simple example is given in the code.
* **loglik_ models_ missing_normalize.py**: In this file, the different likelihood models for the different types of variables considered (real, positive, count, categorical and ordinal) are included.
* **model_ HIVAE_inputDropout.py**: Contains the HI-VAE with input dropout encoder model.
* **model_ HIVAE_factorized.py**: Contains the HI-VAE with factorized encoder model

## Contact

**Alfredo Nazabal**: anazabal@turing.ac.uk