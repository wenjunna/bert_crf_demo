#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22
# @Author  : sunwenjun
# @Site    :
# @File    : model_train.py


# import os
# import sys
# import time
# import glob
import copy
# import shutil
# import random
# import logging
import argparse
# import collections
# from datetime import datetime, timedelta
# from typing import Dict, Optional, Tuple, Union
# import numpy as np
import tensorflow as tf
# from tensorflow.python import keras
# from tensorflow.keras import layers
# from tensorflow.keras import layers, models, Model
# from transformers.modeling_tf_utils import input_processing, TFModelInputType
# from transformers import TFBertPreTrainedModel, BertConfig, TFBertMainLayer
from transformers import BertConfig, TFBertMainLayer
# from models.tf_bert_pretrained_model import TFBertPreTrainedModel
from models.bert_crf import BertCRFModel
from preprocess.ids2tfrecord import SeqLabelDataset
# from models.crf_lib import TFBert
import keras


# keras常用导入方法有以下几种
# import keras
# from tensorflow import keras
# from tensorflow.python import keras
# import tensorflow as tf
# tf.keras


def evaluate_model(model, dev_ds, loss_func):
    it = iter(dev_ds)
    loss_dev = 0
    acc_dev = 0
    count = 0
    try:
        while True:
            data, _ = next(it)

            word_ids = data['word_ids']
            mask_ids = data['mask_ids']
            type_ids = data['type_ids']
            label = data['label']
            text = [word_ids, mask_ids, type_ids]

            higru_ebd_outputs, pred = model(text)
            loss = loss_func(higru_ebd_outputs, label)

            mask = tf.cast(tf.not_equal(word_ids, 0), tf.int32)
            all = tf.reduce_sum(mask)
            pred = pred + -10 * (1 - mask)
            correct = tf.reduce_sum(tf.cast(tf.equal(pred, label), tf.float32))
            loss_dev += loss.numpy()
            acc_dev += (correct / tf.cast(all, tf.float32)).numpy()
            count += 1
    except:
        loss_dev /= count
        acc_dev /= count
        print('dev model\tloss: {}\tacc: {}'.format(
            loss_dev,
            acc_dev
        ), end='')

    return loss_dev, acc_dev


def train_model(model, model_dev, train_ds, dev_ds, loss_func, args):
    opti_bert = keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.95)
    opti_other = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

    # train
    step = 0
    loss_dev_best = 1000
    acc_dev_best = 0

    for i in range(args.epoch):
        loss_train = 0
        acc_train = 0
        it = iter(train_ds)
        try:
            while True:
                data, _ = next(it)
                # text = data['text']
                word_ids = data['word_ids']
                mask_ids = data['mask_ids']
                type_ids = data['type_ids']
                label = data['label']
                text = [word_ids, mask_ids, type_ids]

                params_bert = []
                params_other = []

                with tf.GradientTape() as tape:
                    higru_ebd_outputs, pred = model(text)
                    loss = loss_func(higru_ebd_outputs, label)

                    mask = tf.cast(tf.not_equal(word_ids, 0), tf.int32)
                    all = tf.reduce_sum(mask)
                    pred = pred + -10 * (1 - mask)
                    correct = tf.reduce_sum(tf.cast(tf.equal(pred, label), tf.float32))
                    acc = correct / tf.cast(all, tf.float32)
                    loss_train += loss.numpy()
                    acc_train += (correct / tf.cast(all, tf.float32)).numpy()

                    if (step + 1) % args.print_step == 0:
                        print('epoch:{}\tstep:{}\tloss:{}\tacc:{}'
                              .format(i + 1, step + 1,
                                      loss_train / args.print_step,
                                      acc_train / args.print_step))
                        loss_train = 0
                        acc_train = 0

                    for var in model.trainable_variables:
                        model_name = var.name
                        none_bert_layer = ['tf_bert/bert/pooler/dense/kernel:0',
                                           'tf_bert/bert/pooler/dense/bias:0']

                        if model_name in none_bert_layer:
                            pass
                        elif model_name.startswith('tf_bert'):
                            params_bert.append(var)
                        else:
                            params_other.append(var)

                # 不适用crf的话下面的other参数为空不需要做多余的修改
                params_all = tape.gradient(loss, [params_bert, params_other])
                gradients_bert = params_all[0]
                gradients_other = params_all[1]

                # gradients_other_clipped, norm_other = tf.clip_by_global_norm(gradients_other, 5.0)
                # clip的操作是为了解决梯度爆炸或者消失的问题, 但是在tf2中优化器可以通过衰减速率在控制学习率的变化
                opti_other.apply_gradients(zip(gradients_other, params_other))

                # gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
                opti_bert.apply_gradients(zip(gradients_bert, params_bert))

                step += 1

        except Exception as e:

            print('epoch {} finished'.format(i + 1))
            model.save_weights(args.weight_dir)
            step = 0
            model_dev.load_weights(args.weight_dir)
            loss_dev, acc_dev = evaluate_model(model_dev, dev_ds, loss_func)
            match1 = acc_dev > acc_dev_best
            match2 = acc_dev == acc_dev_best and loss_dev < loss_dev_best

            if match1 or match2:
                acc_dev_best = acc_dev
                loss_dev_best = loss_dev
                print('\t*\tsaving model')
                print('=' * 100)
                tf.saved_model.save(model, args.model_dir)

            else:
                print('')

    # tf.saved_model.save(self.model, save_dir)
    return


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_file", type=str, default="./data/train.tfrecord", help="Train files path")
    parser.add_argument("--dev_file", type=str, default="./data/dev.tfrecord", help="Test files path")
    parser.add_argument("--weight_dir", type=str, default="weights/BertCrfModel/BertCrfModel",
                        help="saving weight path")
    parser.add_argument("--model_dir", type=str, default="./models/bert_crf", help="saving model path")
    parser.add_argument("--n_vocab", type=int, default="21862", help="n_vocab")
    parser.add_argument("--n_label", type=int, default="65", help="n_label")
    parser.add_argument("--seq_length", type=int, default="512", help="seq_length")
    parser.add_argument("--print_step", type=int, default="5", help="print_step")
    parser.add_argument("--epoch", type=int, default="100", help="epoch")
    parser.add_argument("--droprate", type=float, default=0.2, help="droprate")
    parser.add_argument("--batch_size", type=int, default="8", help="batch size")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    data_set = SeqLabelDataset(seq_length=args.seq_length)
    train_ds = data_set.load_tfrecord(args.train_file, args.batch_size)
    dev_ds = data_set.load_tfrecord(args.dev_file, args.batch_size)
    print(next(iter(train_ds)))

    bert_config = BertConfig.from_pretrained("bert-base-chinese", cache_dir="./bert_pretrain")
    bert_config.vocab_size = args.n_vocab
    bert_config.hidden_dropout_prob = 0.5
    bert_config.attention_probs_dropout_prob = 0.5
    bert_config.classifier_dropout = 0.5
    bert_crf_model = BertCRFModel(
        bert_config,
        n_label=args.n_label,
        drop_rate=args.droprate)
    model = bert_crf_model.build(args.seq_length)

    bert_config2 = copy.deepcopy(bert_config)
    bert_config2.hidden_dropout_prob = 0.0
    bert_config2.attention_probs_dropout_prob = 0.0
    bert_config2.classifier_dropout = 0.0
    bert_crf_model2 = BertCRFModel(
        bert_config2,
        n_label=args.n_label,
        drop_rate=0.0)
    model_dev = bert_crf_model2.build(args.seq_length)
    model.summary()

    loss_func = bert_crf_model.get_loss_func()

    # print('inputs: ', [input.op.name for input in model.inputs])
    # print('outputs: ', [output.op.name for output in model.outputs])
    try:
        model.load_weights()
    except:
        pass
    # optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    train_model(model, model_dev, train_ds, dev_ds, loss_func, args)
