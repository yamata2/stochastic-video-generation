import os
import time
import json

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
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file = os.path.join(save_dir, "model.ckpt")

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
    batchsize = train_conf.batchsize        
    height = train_data.shape[2]
    width = train_data.shape[3]
    n_channel = 1

    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)

    global_step = tf.Variable(0, name="global_step", trainable=False)

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
        
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=train_conf.gpu_use_rate),
        device_count={'GPU': 1})
        
    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    error_log = []
    for itr in range(epoch):
        batch_idx = np.random.permutation(8000)[:batchsize]
        batch = train_data[:,batch_idx, :,:]
        batch = np.reshape(batch, list(batch.shape)+[1])
        feed_dict = {is_train: True, x: batch}
        _, latent, recon, total, step = sess.run([train_step,
                                                  latent_loss,
                                                  recon_loss,
                                                  loss,
                                                  global_step],
                                                 feed_dict=feed_dict)
        error_log.append([step, latent, recon, total])
        print "step:{} latent:{}, recon:{}, total:{}".format(step, latent, recon, total)
        if train_conf.test and (step+1) % train_conf.test_interval == 0:
            batch_idx = np.random.permutation(2000)[:batchsize]
            batch = test_data[:,batch_idx, :,:]
            batch = np.reshape(batch, list(batch.shape)+[1])
            feed_dict = {is_train: False,
                         x: batch}
            latent, recon, total = sess.run([latent_loss,
                                             recon_loss,
                                             loss],
                                            feed_dict=feed_dict)
            print "test."
            print "step:{} latent:{}, recon:{}, total:{}".format(step, latent, recon, total)
        if step % train_conf.log_interval == 0:
            saver.save(sess, save_file, global_step=global_step)
    error_log = np.array(error_log)
    np.savetxt("error.log", error_log)

if __name__ == "__main__":
    main()
