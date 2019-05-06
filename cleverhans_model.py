"""
A pure TensorFlow implementation of a convolutional neural network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf

from cleverhans import initializers
from cleverhans.model import Model


class CleverhansModel(Model):
    def __init__(self, scope, nb_classes, my_model, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        # self.nb_filters = nb_filters
    # Do a dummy run of fprop to make sure the variables are created from
    # the start
        # Put a reference to the params in self so that the params get pickled
        self.model = my_model
        self.scope = scope
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs
        # with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
        self.model.build(x)
        return {self.O_LOGITS: self.model.pre_softmax,
                self.O_PROBS: self.model.softmax_layer}

    def make_input_placeholder(self):
        return tf.placeholder(tf.float32, shape = [None, 784])

    def get_params(self):
        """
        Provides access to the model's parameters.
        :return: A list of all Variables defining the model parameters.
        """

        if hasattr(self, 'params'):
            return list(self.params)

        # Catch eager execution and assert function overload.
        try:
            if tf.executing_eagerly():
                raise NotImplementedError("For Eager execution - get_params "
                                      "must be overridden.")
        except AttributeError:
            pass

        # For graph-based execution
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # print(scope_vars)

        if len(scope_vars) == 0:
            self.make_params()
            scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            assert len(scope_vars) > 0

        return scope_vars

    def get_pred(self):
        return self.model.y_pred
