# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Mean
from tensorflow import cast, reshape, float32, float64, int64, constant, concat

from tensorflow.keras.layers import GRU, LayerNormalization, RepeatVector, Dropout, Lambda, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1
from functools import partial
from tensorflow import math
from scipy.stats import bernoulli

def preformat_ori_data(data_preproc, data_imputed):
    
    mask = (np.isnan(data_preproc)).astype('float32') 
    # Basic Parameters
    no, seq_len, dim = np.asarray(data_imputed).shape
    
    # drop square array in favour of ragged array approach 
    list_imputed = list()
    for i in range(no):
        list_imputed.append(data_imputed[i][data_imputed[i].sum(axis = 1) > 0,:])
    list_mask = list()
    for i in range(no):
        list_mask.append(mask[i][data_imputed[i].sum(axis = 1) > 0,:])
    
    # calculate diff time
    for i in range(no):
            list_imputed[i][:,0] = np.append(0, np.diff(list_imputed[i][:,0]))
            
    _dataset = list()
    for i in range(no):
        _dataset.append( np.concatenate( (list_imputed[i] , list_mask[i][:,1:]), axis = 1) )
    
    return _dataset

def post_process_arr(obj):
    no, seq_len, dim = obj.shape
    simulated_data = obj[:,:,0:( tf.cast(1 + dim/2, tf.int32) )]
    simulated_mask = obj[:,:,(tf.cast(1 + dim/2, tf.int32)):dim]
    simulated_mask = bernoulli.rvs(simulated_mask)
    simulated_mask = np.concatenate((np.zeros((no, seq_len, 1)).astype('int64'), simulated_mask), axis = 2)
    simulated_data[simulated_mask == 1] = np.nan
    for i in range(no):
        simulated_data[i,:,0] = np.cumsum(simulated_data[i,:,0]) - simulated_data[i,0,0]
    return simulated_data

def create_recurrent_gan(dim):
    
    RegularisedGRU = partial(GRU, kernel_regularizer=l1(0.01))
    
    def mask_positive_layer(x, dim):
        ones = tf.ones(shape = [tf.cast(1 + (dim / 2), tf.int32) ] ) 
        zeros = tf.zeros(shape = [tf.cast(dim / 2, tf.int32) ])
        mask = tf.concat( [ones, zeros], axis = 0)
        return tf.exp(x) * mask + tf.sigmoid(x) * (1-mask)

    recurrent_generator = Sequential([
        GRU(dim, return_sequences=True, input_shape=(None, dim)),
        RegularisedGRU(dim, return_sequences=True),
        GRU(dim, return_sequences=True),
        Lambda(lambda x: mask_positive_layer(x, dim))
    ])
    
    recurrent_discriminator = Sequential([
        Masking( mask_value= -1.0, input_shape=(None, dim)),
        LayerNormalization( ),
        GRU(dim, return_sequences=True),
        GRU(30, return_sequences=True),
        Dropout(rate=0.2),
        GRU(units = 1, return_sequences=True),
        Lambda(lambda x: tf.sigmoid(x))
    ])
    
    return Sequential([recurrent_generator, recurrent_discriminator])

# Create an optimiser
def create_optimizer():
    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True, clipnorm=1.) 
    return optimizer

def noise_generator(no, seq_len, dim):
    noise = tf.random.normal(shape=[no, seq_len, dim])
    return noise

# define loss
def loss(model, x, target, training = True):
    output = model(x, training = training)
    return binary_crossentropy(y_true = target, y_pred = output)

def grad(model, inputs, targets, mask):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)[mask != 0]
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_gan(dataset, n_epochs, dim):

    # Create model
    optimizer = create_optimizer()
    gan_model = create_recurrent_gan(dim)
    recurrent_generator, recurrent_discriminator = gan_model.layers
    
    # Keep results for plotting
    train_discriminator_loss_results = []
    train_gan_loss_results = []
    
    for epoch in tf.range(n_epochs):
        epoch = cast(epoch, int64)
        epoch_discriminator_loss_avg = Mean()
        epoch_gan_loss_avg = Mean()

        for x_batch, mask_batch in dataset:
            no, seq_len , dim = x_batch.shape
            x_batch = cast(x_batch, float32)
            
            # phase 1 - training the discriminator
            noise = noise_generator(no, seq_len, dim)
            generated_samples = recurrent_generator(noise)
            x_fake_and_real = concat([generated_samples, x_batch], axis=1)
            y1 = cast(reshape(constant([[0.]] * seq_len + [[1.]] * seq_len), [seq_len*2, 1]), float32)
            y1 = tf.broadcast_to(y1, [no, seq_len*2, 1])
            mask1 = tf.ones([no, seq_len])
            mask_fake_and_real = concat([mask1, mask_batch], axis=1)
            recurrent_discriminator.trainable = True
            discriminator_loss_value, discriminator_grads = grad(recurrent_discriminator, x_fake_and_real, y1, mask_fake_and_real) # inputs, target, mask
            optimizer.apply_gradients(zip(discriminator_grads, recurrent_discriminator.trainable_variables))
            
            # phase 2 - training the generator
            noise = noise_generator(no, seq_len, dim)
            y2 = cast(reshape( constant([[1.]] * seq_len), [seq_len, 1]), float32)
            y2 = tf.broadcast_to(y2, [no, seq_len, 1])
            recurrent_discriminator.trainable = False
            gan_loss_value, gan_grads = grad(gan_model, noise, y2, mask1) 
            optimizer.apply_gradients(zip(gan_grads, gan_model.trainable_variables))
            
            # Track progress: Add current batch loss
            epoch_discriminator_loss_avg.update_state(discriminator_loss_value)
            epoch_gan_loss_avg.update_state(gan_loss_value)  
            
        # End epoch
        train_discriminator_loss_results.append(epoch_discriminator_loss_avg.result())
        train_gan_loss_results.append(epoch_gan_loss_avg.result())
        
        if epoch % 50 == 0:
            print("Epoch {:03d}: Discriminator Loss: {:.3f}".format(epoch, epoch_discriminator_loss_avg.result() ) , file=sys.stdout)
            print("Epoch {:03d}: GAN Loss: {:.3f}".format(epoch, epoch_gan_loss_avg.result() ) , file=sys.stdout)
            
    return gan_model, train_discriminator_loss_results, train_gan_loss_results

def fit_recurrent_gan(data_preproc, data_imputed):
    
    num_epochs = 2001
    batch_size = 128
    
    print('Preprocessing data ...', file=sys.stdout)
    # convert ragged -> uniform
    formated_data = preformat_ori_data(data_preproc, data_imputed)
    dim = formated_data[0].shape[1]
    ds = tf.ragged.constant(formated_data)
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.map(lambda l: l.to_tensor()) 
    mask_value = cast(-1, float64)
    ds = ds.padded_batch(batch_size, padded_shapes=[None, dim], padding_values= mask_value ).prefetch(1)  
    ds = tf.data.Dataset.zip((ds.map(lambda x: cast(x , float32)), # x_batch
                              ds.map(lambda m: cast(m[:,:,0] != mask_value , float32)) # Mask
                             ))
    print('Starting recurrent GAN ...', file=sys.stdout)
    # Train model
    recurrent_gan,  _train_discriminator_loss, _train_gan_loss = train_gan(ds, num_epochs, dim)
    
    # Generate output data
    recurrent_generator, recurrent_discriminator = recurrent_gan.layers
    noise = noise_generator(preproc_data.shape)
    generated_samples = recurrent_generator(noise)
    
    return post_process_arr( generated_samples.numpy() )
    
  