#!/usr/bin/env python
# coding: utf-8

# In[54]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import vae


# In[94]:


model_dir = 'models/cvae03'
train_path = "./features/train/03/feature_attributions.npy"
# Setting up the data and the model
feature_attributions = np.load(train_path)
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
labels = mnist.train.labels
train_total_data = np.column_stack((feature_attributions,labels))


# In[95]:


np.max(labels - train_total_data[:,-1])


# In[96]:


dim_x = 784
dim_y = 10
dim_z = 2
n_hidden = 500
learn_rate = 1e-3

num_examples = mnist.train.num_examples
n_epochs = 20
batch_size = 50
total_batch = int(num_examples / batch_size)

ADD_NOISE = False


# In[98]:


tf.reset_default_graph() 
""" build graph """
# input placeholders
# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
x_hat = tf.placeholder(tf.float32, shape=[None, dim_x], name='input_img')
x = tf.placeholder(tf.float32, shape=[None, dim_x], name='target_img')
y = tf.placeholder(tf.int32, shape=[None], name='target_labels')
y_one_hot = tf.one_hot(indices = y, depth=dim_y)

# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
fack_id_in = tf.placeholder(tf.float32, shape=[None, dim_y], name='latent_variable') # condition


# network architecture
x_, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(x_hat, x, y_one_hot, dim_x, dim_z, n_hidden, keep_prob)

global_step = tf.contrib.framework.get_or_create_global_step()

# optimization
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step=global_step)

saver = tf.train.Saver(max_to_keep=3)


# In[105]:


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})
    
    for epoch in range(n_epochs):
        # Random shuffling
        np.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-1]
        train_labels_ = train_total_data[:, -1]

        for i in range(total_batch):

            offset = (i * batch_size) % (num_examples)
            batch_xs_input = train_data_[offset:(offset + batch_size), :]
            batch_ys_input = train_labels_[offset:(offset + batch_size)]
            batch_xs_target = batch_xs_input

            # add salt & pepper noise
            if ADD_NOISE:
                batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

            _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                (train_op, loss, neg_marginal_likelihood, KL_divergence),
                feed_dict={x_hat: batch_xs_input, x: batch_xs_target, y: batch_ys_input, keep_prob : 0.9})

            if (i+epoch*total_batch) % 300 == 0:
                print("training step %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (i+epoch*total_batch, tot_loss, loss_likelihood, loss_divergence))
                saver.save(sess,
                         os.path.join(model_dir, 'checkpoint'),
                         global_step=global_step)

