#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
This tutorial shows how to generate adversarial examples
using JSMA in white-box setting.
The original paper can be found at:
https://arxiv.org/abs/1511.07528
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.loss import CrossEntropy
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import DeepFool
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_eval, model_argmax

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from cleverhans_model import CleverhansModel
from model import Model as My_model

from tensorflow.python import pywrap_tensorflow
import math
from random import choice

import matplotlib.pyplot as plt


# In[2]:


model_dir = "./models/nat"

num_examples = 10000
batch_size = 200
num_batches = int(math.ceil(num_examples / batch_size))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_test = mnist.test.images
y_test = mnist.test.labels


# In[3]:


def classify(sess, img):
    x = tf.placeholder(tf.float32, shape = [None, 784])
    c_model.fprop(x)
    return sess.run(c_model.get_pred(), feed_dict = {x: img})
def plot(img):
    plt.imshow(np.resize(img,[28,28]), cmap='Greys_r')


# In[4]:


def init_graph(model_dir):
    tf.reset_default_graph()
    # print(scope_vars)
    # x = tf.placeholder(tf.float32, shape = [None, 784])

    model = My_model()
    c_model = CleverhansModel('CNN', 10, model)
    checkpoint = tf.train.latest_checkpoint(model_dir)
    reader=pywrap_tensorflow.NewCheckpointReader(checkpoint)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.log_device_placement=False
    config.allow_soft_placement=True
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint)
    return session, c_model

def batch_attack(data, attack, **params):
    x_adv = []
    for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, num_examples)
        print('batch size: {}'.format(bend - bstart))

        x_batch = data[bstart:bend, :]

        x_batch_adv = attack.generate_np(x_batch, **params)
        
        x_adv = x_adv + x_batch_adv.tolist()
        
    return x_adv

def save_npy(algm, x_adv, filename = '/adv.npy'):
    path = './adv_test/' + algm + filename
    x_adv_np = np.asarray(x_adv)
    np.save(path, x_adv_np)
    print("saved to: " + path)
    


# In[5]:


session, c_model = init_graph(model_dir)


# In[6]:


###########################################################################
# Craft adversarial examples using the Jacobian-based saliency map approach
###########################################################################

# y = tf.placeholder(tf.int64, shape = [None])
print("=============================================================")
print('JSMA: Crafting ' + str(num_examples) +
    ' adversarial examples')

jsma = SaliencyMapMethod(c_model, sess=session)
jsma_params = {'theta': 1., 'gamma': 0.1,
             'clip_min': 0., 'clip_max': 1.,
             'y_target': None}

x_adv = [] # adv accumulator

for sample_ind in xrange(0, num_examples):
    if(sample_ind % batch_size == 0):
        print('Attacking input batch %i/%i' % (sample_ind/batch_size + 1, num_examples/batch_size))
    sample = x_test[sample_ind:(sample_ind + 1)]
    adv = jsma.generate_np(sample, **jsma_params)
    x_adv.append(adv)
    
x_adv = np.concatenate(x_adv, axis=0)
save_npy('jsma', x_adv)
session.close()


# In[7]:


session, c_model = init_graph(model_dir)


# In[8]:


LEARNING_RATE = .001
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 100

print('Iterating over {} batches'.format(num_batches))

# x = tf.placeholder(tf.float32, shape = [None, 784])
# y = tf.placeholder(tf.int64, shape = [None])

###########################################################################
# Craft adversarial examples using Carlini and Wagner's approach
###########################################################################
print("=============================================================")
print('CW: Crafting ' + str(num_examples) + ' adversarial examples')
print("This could take some time ...")

# Instantiate a CW attack object
cw = CarliniWagnerL2(c_model, sess=session)

cw_params = {'binary_search_steps': 1,
               "y_target": None,
               'max_iterations': ATTACK_ITERATIONS,
               'learning_rate': CW_LEARNING_RATE,
               'batch_size': batch_size,
               'initial_const': 10}

x_adv = batch_attack(x_test, cw, **cw_params)
save_npy('cw', x_adv)
session.close()


# In[9]:


session, c_model = init_graph(model_dir)


# In[10]:


fgsm_params = {
  'eps': 0.3,
  'clip_min': 0.,
  'clip_max': 1.
}

# Set TF random seed to improve reproducibility
tf.set_random_seed(2333)

# x = tf.placeholder(tf.float32, shape = [None, 784])
# y = tf.placeholder(tf.int64, shape = [None])

###########################################################################
# Craft adversarial examples using FGSM
###########################################################################
print("=============================================================")
print('FGSM: Crafting ' + str(num_examples) + ' adversarial examples')

fgsm = FastGradientMethod(c_model, sess=session)
x_adv = batch_attack(x_test, fgsm, **fgsm_params)
save_npy('fgsm', x_adv)
session.close()


# In[11]:


session, c_model = init_graph(model_dir)


# In[12]:



###########################################################################
# Craft adversarial examples using FGSM
###########################################################################
print("DEEPFOOL: =============================================================")
print('Crafting ' + str(num_examples) + ' adversarial examples')

deepfool = DeepFool(c_model, sess=session)
x_adv = batch_attack(x_test, deepfool)
save_npy('deepfool', x_adv)

session.close()


# In[13]:


def init_my_model(model_dir):
    tf.reset_default_graph()
    # print(scope_vars)
    x = tf.placeholder(tf.float32, shape = [None, 784])
    y = tf.placeholder(tf.int64, shape = [None])
    
    model = My_model()
    model.build_and_eval(x, y)
    
    checkpoint = tf.train.latest_checkpoint(model_dir)
    reader=pywrap_tensorflow.NewCheckpointReader(checkpoint)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.log_device_placement=False
    config.allow_soft_placement=True
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint)
    return session, model

def generate_adv_attr(session, model, img_path, tar_path, num_examples = num_examples):

    
    
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


# In[1]:



adv_dir = "./adv_test/"
tar_dir = "./features/test/"

algm = ['/fgsm/','/cw/','/jsma/','/deepfool/']
# loss = ['/xent/', '/cw/']
model_name = '/nat/'
name = 'adv.npy'
name_out = 'adv_attr.npy'

# In[2]:



# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
# labels = mnist.test.labels

for a in range(len(algm)):
    print("=============================================================")
    print("model_dir:" + model_dir)
    session, model = init_my_model(model_dir)
    
    adv_path = adv_dir + algm[a] + name
    tar_path = tar_dir + algm[a] + name_out

    print("adv_path:" + adv_path)
    generate_adv_attr(session, model, adv_path, tar_path)
    session.close()


# In[ ]:




