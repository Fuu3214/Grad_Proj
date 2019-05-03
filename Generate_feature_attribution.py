#!/usr/bin/env python
# coding: utf-8

# In[84]:


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


# In[85]:


model_dir = "./models/nat"


# In[86]:


tf.reset_default_graph()
model = Model()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.log_device_placement=False
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())


checkpoint = tf.train.latest_checkpoint(model_dir)
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint)
saver.restore(session, checkpoint)


# In[87]:


mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
num_examples = mnist.train.num_examples


# In[88]:


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


# In[116]:


feature_attributions = []
for i in range(num_examples):
    x = mnist.train.images[i]
    y = mnist.train.labels[i]
    feature_attributions.append(integrated_gradient(x, [y]))


# In[117]:


print('Storing examples')
path = "./features/nat/feature_attributions.npy"
feature_attributions = np.asarray(feature_attributions)
np.save(path, feature_attributions)
print('Examples stored in {}'.format(path))


# In[118]:


features = np.load(path)


# In[119]:


features.shape


# In[ ]:




