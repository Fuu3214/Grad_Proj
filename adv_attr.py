#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Evaluates feature attributes of a adversarialy trained model against adv attack"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import PIL
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np

from model import Model

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops.parallel_for.gradients import jacobian


# In[20]:


def init_sess(model_path):
    tf.reset_default_graph()
    model = Model()
    checkpoint = tf.train.latest_checkpoint(model_path)
    reader=pywrap_tensorflow.NewCheckpointReader(checkpoint)
    saver = tf.train.Saver()


    # In[4]:


    config = tf.ConfigProto()
    config.log_device_placement=False
    config.allow_soft_placement=True
    config.gpu_options.allow_growth=True

    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint)
    return session, model


    # In[15]:

# In[3]:


def generate_adv_attr(session, model, img_path, tar_path, num_examples = 10000):

    
    
    adv_test = np.load(img_path)

    # train_total_data = np.column_stack((adv_test,true_labels))


    # In[16]:


    labeled_pred = model.softmax_layer[:,model.y_input[0]]
    grad = tf.gradients(labeled_pred, model.x_input)
    def integrated_gradient(img, target_label_index, steps = 50, baseline=None):
        if baseline is None:
            baseline = 0*img
        assert(baseline.shape == img.shape)
        steps=steps

        # Scale input and compute gradients.
        scaled_inputs = [baseline + (float(i)/steps)*(img-baseline) for i in range(0, steps+1)]

        gradient = session.run(grad, feed_dict = {model.x_input:np.squeeze(scaled_inputs),model.y_input:target_label_index})
        avg_grads = np.average(gradient[0][:-1], axis=0)
        integrated_gradients = (img-baseline)*avg_grads  # shape: <inp.shape>
        return integrated_gradients


    # In[17]:


    feature_attributions = []
    for i in range(num_examples):
        x = adv_test[i]
        y = session.run(model.y_pred, feed_dict={model.x_input: [x]})[0]
        feature_attributions.append(integrated_gradient(x, [y]))


    print('Storing examples')
    feature_attributions = np.asarray(feature_attributions)
    np.save(tar_path, feature_attributions)
    print('Examples stored in {}'.format(tar_path))



model_dir = "./models/"
adv_dir = "./adv_test/"
tar_dir = "./features/test/"

algm = ['/pgd/']
loss = ['/xent/', '/cw/']
model_name = ['/nat/', '/01/', '/02/']
name = ['/01.npy','/02.npy','/03.npy']

# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
# labels = mnist.test.labels

for a in range(len(algm)):
    a_path = algm[a]
    for l in range(len(loss)):
        l_path = a_path + loss[l]
        for m in range(len(model_name)):
            m_path = l_path + model_name[m]
            model_path = model_dir + model_name[m]

            session, model = init_sess(model_path)

            print("=================")
            print("model_path:" + model_path)
            for n in range(len(name)):
                n_path = m_path + name[n]
                adv_path = adv_dir + n_path
                tar_path = tar_dir + n_path
                
                print("adv_path:" + adv_path)
                generate_adv_attr(session, model, adv_path, tar_path)
            session.close()

