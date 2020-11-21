"""
The hider module containing the `hider(...)` function.
"""
# pylint: disable=fixme
from typing import Dict, Union, Tuple, Optional
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Mean
from tensorflow import cast, reshape, float32, float64, int64, constant, concat
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Lambda, GRU, LayerNormalization, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from functools import partial
from scipy.stats import bernoulli
from utils.data_preprocess import preprocess_data

# ---------------------------- #
# GAIN Imputation network
# ---------------------------- #

def hint_generator(X, M):
    # takes X: data with raw imputed values
    #       M: mask
    # return : hint input to generator network.
    noise =  tf.random.normal(shape=X.shape, stddev=1) 
    hint = tf.keras.activations.relu( M * (X + noise) + (1 - M) * X ) 
    return hint

def gain_loss(model, x, target, training = True):
    output = model(x, training = training)
    return binary_crossentropy(y_true = target, y_pred = output)

def gain_grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = gain_loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def create_gain(dim):
    gain_generator = Sequential([
        BatchNormalization(input_shape = [dim*2]),
        Dense(100, activation = "relu"),
        BatchNormalization(),
        Dense(150, activation = "relu"),
        BatchNormalization(),
        Dense(dim, activation = "relu")
    ])
    gain_discriminator = Sequential([
        BatchNormalization(input_shape = [dim]),
        Dense(150, activation = "relu", input_shape = [dim]),
        BatchNormalization(),
        Dense(100, activation = "relu"),
        BatchNormalization(),
        Dense(dim, activation = "sigmoid")
    ])
    return Sequential([gain_generator, gain_discriminator])


def gain_train_step(dataset, gain, n_epochs):
    generator, discriminator = gain.layers
    discriminator_optimizer = keras.optimizers.SGD(momentum=0.9, nesterov=True) 
    generator_optimizer = keras.optimizers.Adam()
    # Keep results for plotting
    train_discriminator_loss_results = []
    train_generator_loss_results = []
    
    for epoch in range(n_epochs):
        epoch_discriminator_loss_avg = Mean()
        epoch_generator_loss_avg = Mean()
        for x_batch, mask_batch in dataset:
            x_batch = cast(x_batch, float32)
            mask_batch = cast(mask_batch, float32)
            # phase 1: train discriminator
            hint = hint_generator(x_batch, mask_batch)
            generated_samples = generator(concat( [hint, mask_batch], axis = 1))
            discriminator.trainable = True
            discriminator_loss_value, discriminator_grads = gain_grad(discriminator, generated_samples, mask_batch) 
            discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))
            # phase 2 - training the generator
            hint = hint_generator(x_batch, mask_batch)
            discriminator.trainable = False
            generator_loss_value, generator_grads = gain_grad(gain, concat( [hint, mask_batch], axis = 1), mask_batch) 
            generator_optimizer.apply_gradients(zip(generator_grads, gain.trainable_variables))
            # Track progress: Add current batch loss
            epoch_discriminator_loss_avg.update_state(discriminator_loss_value)
            epoch_generator_loss_avg.update_state(generator_loss_value)
            
        # End epoch
        train_discriminator_loss_results.append(epoch_discriminator_loss_avg.result())
        train_generator_loss_results.append(epoch_generator_loss_avg.result())
        if epoch % 50 == 0:
            print("GAIN Epoch {:03d}: Discriminator Loss: {:.3f}".format(epoch, epoch_discriminator_loss_avg.result() ) , file=sys.stdout)
            print("GAIN Epoch {:03d}: Generator Loss: {:.3f}".format(epoch, epoch_generator_loss_avg.result() ) , file=sys.stdout)
            
    return gain, train_discriminator_loss_results, train_generator_loss_results


def imputation_gain(data_preproc, data_imputed, n_epochs):
    # set batch size for gain
    batch_size = 128
    # prepare data for gain
    data_imputed, data_mask, seq_times, max_seq_len = preformat_data(data_preproc, data_imputed)
    data_imputed = np.concatenate(data_imputed, axis = 0)
    data_mask = np.concatenate(data_mask, axis = 0)
    data_len, dim = data_imputed.shape
    ds = tf.data.Dataset.from_tensor_slices((data_imputed, data_mask))
    ds = ds.batch(batch_size).prefetch(1)
    # train gain model
    gain_model = create_gain(dim)
    gain_model, _discriminator, _generator = gain_train_step(ds, gain_model, n_epochs)
    generator , discriminator = gain_model.layers
    # generate imputed samples
    hint = hint_generator(data_imputed, data_mask)
    gain_data_imputed = data_mask * generator(concat( [hint, data_mask], axis = 1)) + (1 - data_mask) * data_imputed
    gain_data_imputed = post_process_gain(gain_data_imputed, data_mask, seq_times)
    
    return gain_data_imputed, max_seq_len

# ----------------------------- #
# AZWarwick hider: recurrent GAN for real value positive data
# Guanya
# ----------------------------- #

def create_recurrent_gan(dim):
    RegularisedGRU = partial(GRU, kernel_regularizer=l2(0.01))
    def mask_positive_layer(x, dim):
        ones = tf.ones(shape = [tf.cast(1 + (dim / 2), tf.int32) ] ) 
        zeros = tf.zeros(shape = [tf.cast(dim / 2, tf.int32) ])
        mask = tf.concat( [ones, zeros], axis = 0)
        return tf.keras.activations.relu(x + 1) * mask + tf.sigmoid(x) * (1-mask)
    recurrent_generator = Sequential([
        RegularisedGRU(100, return_sequences=True, input_shape=(None, dim)),
        LayerNormalization( ),
        RegularisedGRU(120, return_sequences=True),
        LayerNormalization( ),
        RegularisedGRU(150, return_sequences=True),
        Dropout(rate=0.2),
        RegularisedGRU(dim, return_sequences=True),
        Lambda(lambda x: mask_positive_layer(x, dim))
    ])
    recurrent_discriminator = Sequential([
        Masking( mask_value= -1.0, input_shape=(None, dim)),
        LayerNormalization( ),
        RegularisedGRU(150, return_sequences=True),
        LayerNormalization( ),
        RegularisedGRU(120, return_sequences=True),
        LayerNormalization( ),
        RegularisedGRU(100, return_sequences=True),
        Dropout(rate=0.2),
        RegularisedGRU(units = 1, return_sequences=True),
        Lambda(lambda x: tf.sigmoid(x))
    ])
    return Sequential([recurrent_generator, recurrent_discriminator])

def noise_generator(no, seq_len, dim):
    noise = tf.random.normal(mean=20.0, stddev=4.0, shape=[no, seq_len, dim])
    return tf.keras.activations.relu(noise)

def loss(model, x, target, training = True):
    output = model(x, training = training)
    return binary_crossentropy(y_true = target, y_pred = output)

def grad(model, inputs, targets, mask):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)[mask != 0]
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_rgan(gan_model, dataset, n_epochs):
    
    generator_optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, clipnorm=1.) 
    discriminator_optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True, clipnorm=1.) 
    recurrent_generator, recurrent_discriminator = gan_model.layers
    
    # Keep results for plotting
    train_discriminator_loss_results = []
    train_generator_loss_results = []
    
    for epoch in range(n_epochs):
        epoch_discriminator_loss_avg = Mean()
        epoch_generator_loss_avg = Mean()

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
            discriminator_loss_value, discriminator_grads = grad(recurrent_discriminator, x_fake_and_real, y1, mask_fake_and_real) 
            discriminator_optimizer.apply_gradients(zip(discriminator_grads, recurrent_discriminator.trainable_variables))
            # phase 2 - training the generator
            noise = noise_generator(no, seq_len, dim)
            y2 = cast(reshape( constant([[1.]] * seq_len), [seq_len, 1]), float32)
            y2 = tf.broadcast_to(y2, [no, seq_len, 1])
            recurrent_discriminator.trainable = False
            generator_loss_value, generator_grads = grad(gan_model, noise, y2, mask1) 
            generator_optimizer.apply_gradients(zip(generator_grads, gan_model.trainable_variables))
            # Track progress: Add current batch loss
            epoch_discriminator_loss_avg.update_state(discriminator_loss_value)
            epoch_generator_loss_avg.update_state(generator_loss_value)  
            
        # End epoch
        train_discriminator_loss_results.append(epoch_discriminator_loss_avg.result())
        train_generator_loss_results.append(epoch_generator_loss_avg.result())
        
        if epoch % 50 == 0:
            print("RGAN Epoch {:03d}: Discriminator Loss: {:.3f}".format(epoch, epoch_discriminator_loss_avg.result() ) , file=sys.stdout)
            print("RGAN Epoch {:03d}: Generator Loss: {:.3f}".format(epoch, epoch_generator_loss_avg.result() ) , file=sys.stdout)
            
    return gan_model, train_discriminator_loss_results, train_generator_loss_results

def fit_recurrent_gan(imputed_data, max_length, num_epoch, batch_size):
    dim = imputed_data[0].shape[1]
    no = len(imputed_data)
    imputed_data = pre_process_time(imputed_data)
    ds = tf.ragged.constant(imputed_data)
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.map(lambda l: l.to_tensor()) 
    mask_value = cast(-1, float32)
    ds = ds.padded_batch(batch_size, padded_shapes=[None, dim], padding_values= mask_value ).prefetch(1)  
    ds = tf.data.Dataset.zip((ds.map(lambda x: cast(x , float32)), # x_batch
                              ds.map(lambda m: cast(m[:,:,0] != mask_value , float32)) # Mask
                             ))
    gan_model = create_recurrent_gan(dim)
    # Train model
    gan_model,  _discriminator_loss, _generator_loss = train_rgan(gan_model, ds, num_epoch)
    
    # Generate output data
    recurrent_generator, recurrent_discriminator = gan_model.layers
    noise = noise_generator(no, max_length, dim)
    generated_samples = recurrent_generator(noise)
    generated_samples = post_process_rgan( generated_samples.numpy() )
    return generated_samples, gan_model, _discriminator_loss, _generator_loss

# ----------------------------- #
# Data helpers 
# ----------------------------- #
def preformat_data(data_preproc, data_imputed):
    mask = (np.isnan(data_preproc)).astype('float32') 
    # Basic Parameters
    no, seq_len, dim = np.asarray(data_imputed).shape
    # drop rectangular array in favour of ragged array approach 
    foo_imputed = list()
    for i in range(no):
        foo_imputed.append(data_imputed[i][data_imputed[i].sum(axis = 1) > 0,:])
    foo_mask = list()
    for i in range(no):
        foo_mask.append(mask[i][data_imputed[i].sum(axis = 1) > 0,:])
    get_seq_len = lambda l: l.shape[0]
    seq_times = list(map(get_seq_len, foo_imputed)) 
    return foo_imputed, foo_mask, seq_times, seq_len

def post_process_gain(generated_imputed_data, mask, seq_times):
    no = len(seq_times)
    foo_generated_imputed = []
    ind = 0
    for i in range(no):
        foodat = generated_imputed_data[ind:(seq_times[i] + ind)]
        foomask =  mask[ind:(seq_times[i] + ind)]
        foo_generated_imputed.append(  np.concatenate( (foodat, foomask[:,1:]), axis = 1) )
        ind = seq_times[i] + ind
    return foo_generated_imputed

def pre_process_time(obj):
    no = len(obj)
    for i in range(no):
        obj[i][:,0] = np.append(0, np.diff(obj[i][:,0]))
    return obj

def post_process_rgan(obj):
    no, seq_len, dim = obj.shape
    simulated_data = obj[:,:,0:( tf.cast(1 + dim/2, tf.int32) )]
    simulated_mask = obj[:,:,(tf.cast(1 + dim/2, tf.int32)):dim]
    simulated_mask = bernoulli.rvs(simulated_mask)
    simulated_mask = np.concatenate((np.zeros((no, seq_len, 1)).astype('int64'), simulated_mask), axis = 2)
    simulated_data[simulated_mask == 1] = np.nan
    for i in range(no):
        simulated_data[i,:,0] = np.cumsum(simulated_data[i,:,0]) - simulated_data[i,0,0]
    return simulated_data


def hider(input_dict: Dict) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Solution hider function.

    Args:
        input_dict (Dict): Dictionary that contains the hider function inputs, as below:
            * "seed" (int): Random seed provided by the competition, use for reproducibility.
            * "data" (np.ndarray of float): Input data, shape [num_examples, max_seq_len, num_features].
            * "padding_mask" (np.ndarray of bool): Padding mask of bools, same shape as data.

    Returns:
        Return format is:
            np.ndarray (of float) [, np.ndarray (of bool)]
        first argument is the hider generated data, expected shape [num_examples, max_seq_len, num_features]);
        second optional argument is the corresponding padding mask, same shape.
    """

    # Get the inputs.
    seed = input_dict["seed"]  # Random seed provided by the competition, use for reproducibility.
    data = input_dict["data"]  # Input data, shape [num_examples, max_seq_len, num_features].
    padding_mask = input_dict["padding_mask"]  # Padding mask of bools, same shape as data.

    # Get processed and imputed data:
    data_preproc, data_imputed = preprocess_data(data, padding_mask)
    
    print('Starting imputation GAIN ...')
    num_epoch = 4000
    batch_size = 64
    data_imputed, max_length = imputation_gain(data_preproc, data_imputed, num_epoch)
    
    print('Starting recurrent GAN ...', file=sys.stdout)
    generated_samples, gan_model, _discriminator_loss, _generator_loss = fit_recurrent_gan(data_imputed, max_length, num_epoch, batch_size)
    
    return generated_samples


