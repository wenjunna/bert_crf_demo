#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 3:35 PM
# @Author  : sunwenjun
# @File    : crf_layer.py
# @brief: PyCharm
import tensorflow as tf
from models import crf_lib
import keras


class CRFLayer(keras.layers.Layer):

    def __init__(self, num_tags=6,
                 transition_params=None,
                 **kwargs):
        super(CRFLayer, self).__init__(**kwargs)
        self.num_tags = num_tags

        if transition_params is None:
            initializer = keras.initializers.GlorotUniform()
            self.transition_params = tf.Variable(
                initializer([num_tags, num_tags]))
        else:
            self.transition_params = transition_params

    def build(self, input_shape):
        super(CRFLayer, self).build(input_shape)
        print("build CRFLayer layer success!")

    def call(self, inputs, training=False):
        used = tf.sign(tf.abs(inputs))
        sequence_lengths = tf.reduce_mean(tf.reduce_sum(used, axis=1), axis=1)
        sequence_lengths = tf.cast(sequence_lengths, tf.int64)

        decode_tags, best_score = crf_lib.crf_decode(
            inputs,
            self.transition_params,
            sequence_lengths
        )
        return decode_tags

    def loss(self, inputs, label):
        used = tf.sign(tf.abs(inputs))
        sequence_lengths = tf.reduce_mean(tf.reduce_sum(used, axis=1), axis=1)
        sequence_lengths = tf.cast(sequence_lengths, tf.int64)

        log_likelihood, transition_params = crf_lib.crf_log_likelihood(
            inputs,
            label,
            sequence_lengths,
            self.transition_params
        )
        cost = tf.reduce_mean(-log_likelihood)
        return cost

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(CRFLayer, self).get_config()
        return base_config
