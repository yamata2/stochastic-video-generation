import os
import time
import json

import numpy as np
import tensorflow as tf
lstm = tf.contrib.rnn.LSTMCell
fc = tf.contrib.layers.fully_connected

from layers import full_connection, convolution2d, transposed_convolution2d, reshape

from IPython import embed

def make_cells(n_layers, n_units):
    cells = []
    for j in range(n_layers):
        with tf.variable_scope("layer_{}".format(j), reuse=False):
            cell = lstm(num_units=n_units,
                        use_peepholes=True,
                        forget_bias=0.8)
            cells.append(cell)
    return tf.contrib.rnn.MultiRNNCell(cells)

def image_processor(h, layers):
    functions = {"fc": full_connection,
                 "conv": convolution2d,
                 "deconv": transposed_convolution2d,
                 "reshape": reshape}
    phase = tf.constant(True)
    with tf.variable_scope("visual_feature_extractor"):
        for l in range(len(layers)):
            scope = "layer%.2d" %l
            layer = layers[scope]
            h = functions[layer["type"]](h, layer["param"], phase, scope)
            print h
    return h

def inference(x, n_units=128, n_layers=1, n_out=10, scope="Inference"):
    """Infer a posterior approximately. 

    Returns:
        means: [data_len, batchsize, n_out]
               means of Gaussian distribution.
        log_var: [data_len, batchsize, n_out]
               std_dev of Gaussian distribution.
    """

    means = []
    log_vars = []
    with tf.variable_scope(scope, reuse=False):
        cell = make_cells(n_layers, n_units)
        state = cell.zero_state(x.get_shape().as_list()[1], tf.float32)
        for i in range(x.shape[0]):
            if i == 0:
                fc_reuse = False
            else:
                tf.get_variable_scope().reuse_variables()
                fc_reuse = True
            h, state = cell(x[i], state)
            mean = fc(h, n_out,
                      activation_fn=None,
                      reuse=fc_reuse,
                      scope="mean")
            log_var= fc(h, n_out,
                        activation_fn=None,
                        reuse=fc_reuse,
                        scope="log_var")
            means.append(mean)
            log_vars.append(log_var)

    return tf.stack(means), tf.stack(log_vars)

def generation(x, z_mean, z_log_var,
               n_units, n_layers, n_out, scope="Generation"):
    """Generate frames.

    Return:
        y: [data_len, batchsize, n_out]
           n_out is the size of h (i.e., before decoding)
    """
    # Draw one sample z from Gaussian distribution
    eps = tf.random_normal(z_mean.get_shape().as_list(),
                           0, 1, dtype=tf.float32)
    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_var)), eps))

    y = []
    with tf.variable_scope(scope, reuse=False):
        cell = make_cells(n_layers, n_units)
        state = cell.zero_state(x.get_shape().as_list()[1], tf.float32)
        for i in range(x.shape[0]):
            if i == 0:
                fc_reuse = False
                current_x = x[i]
            else:
                tf.get_variable_scope().reuse_variables()
                fc_reuse = True
                current_x = x[i-1]
            current_in = tf.concat([current_x, z[i]], axis=1)
            h, state = cell(current_in, state)
            out = fc(h, n_out,
                     activation_fn=tf.nn.tanh,
                     reuse=fc_reuse,
                     scope="out")
            y.append(out)
    return tf.stack(y)
