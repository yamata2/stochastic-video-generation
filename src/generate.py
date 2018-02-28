import os
import time
import json
import cv2

import numpy as np
import tensorflow as tf
from IPython import embed

from config import NetConfig, TrainConfig
from modules import *

def main():
    encoder_conf = open("./encoder.json", "r")
    encoder_layers = json.load(encoder_conf)
    decoder_conf = open("./decoder.json", "r")
    decoder_layers = json.load(decoder_conf)

    net_conf = NetConfig()
    net_conf.set_conf("./net_conf.txt")
    q_units = net_conf.inference_num_units
    q_layers = net_conf.inference_num_layers
    p_units = net_conf.prediction_num_units
    p_layers = net_conf.prediction_num_layers
    latent_dim = net_conf.latent_dim
    beta = net_conf.regularize_const
    
    train_conf = TrainConfig()
    train_conf.set_conf("./train_conf.txt")
    seed = train_conf.seed
    epoch = train_conf.epoch
    save_dir = train_conf.save_dir
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))

    data_file = train_conf.data_file
    if not os.path.exists(data_file):
        if not os.path.exists(os.path.dirname(data_file)):
            os.mkdir(os.path.dirname(data_file))
        import wget
        url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
        wget.download(url, out=data_file)
    data = np.load(data_file)
    data = (data / 255.) * 0.9 + 0.05
    train_data = data[:, :8000, :, :]
    test_data = data[:, 8000:, :, :]

    data_len = train_data.shape[0]
    batchsize = 1
    height = train_data.shape[2]
    width = train_data.shape[3]
    n_channel = 1

    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)

    x = tf.placeholder(tf.float32,
                       [data_len, batchsize,
                        height, width, n_channel])
    reshaped_x = tf.reshape(x, [-1, height, width, n_channel])

    with tf.variable_scope("encoder", reuse=False):
        h = image_processor(reshaped_x, encoder_layers)
    h = tf.reshape(h, [data_len, batchsize, -1])

    z_post_mean, z_post_log_var = inference(h,
                                            q_units,
                                            q_layers,
                                            latent_dim)
    z_prior_mean = tf.zeros(z_post_mean.get_shape().as_list(), tf.float32)
    z_prior_log_var = tf.ones(z_post_log_var.get_shape().as_list(), tf.float32)

    is_train = tf.placeholder(tf.bool)
    z_mean = tf.cond(is_train, lambda: z_post_mean, lambda: z_prior_mean)
    z_log_var = tf.cond(is_train, lambda: z_post_log_var, lambda: z_prior_log_var)
    g = generation(h, z_mean, z_log_var,
                   p_units, p_layers, int(h.get_shape()[2])) 
    g = tf.reshape(g, [-1, int(h.get_shape()[2])])
    with tf.variable_scope("decoder", reuse=False):
        reshaped_y = image_processor(g, decoder_layers)
    y = tf.reshape(reshaped_y, x.get_shape().as_list())
    with tf.name_scope('loss'):
        # mse is ok? not cross entropy?
        recon_loss = tf.reduce_mean(tf.square(x - y))
        latent_loss = -tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        # what is the appropriate beta...
        loss = recon_loss + beta * latent_loss
        
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=train_conf.gpu_use_rate),
        device_count={'GPU': 1})
        
    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, save_dir)

    error_log = []
    for i in range(train_data.shape[1]):
        print "train {}".format(i)
        batch = train_data[:,i:i+1, :,:]
        batch = np.reshape(batch, list(batch.shape)+[1])
        feed_dict = {is_train: True,
                     x: batch}
        outputs = sess.run(y, feed_dict=feed_dict)
        save_as_images(outputs, i, "train")

    for i in range(test_data.shape[1]):
        print "test {}".format(i)
        batch = test_data[:,i:i+1, :,:]
        batch = np.reshape(batch, list(batch.shape)+[1])
        feed_dict = {is_train: True,
                     x: batch}
        outputs = sess.run(y, feed_dict=feed_dict)
        save_as_images(outputs, i, "test")

def save_as_images(array, index, base):
    dirname = os.path.join(base, "%.6d" %index)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for i in range(array.shape[0]):
        cv2.imwrite(dirname+"/%.3d.png" %i, array[i,0])

if __name__ == "__main__":
    main()
