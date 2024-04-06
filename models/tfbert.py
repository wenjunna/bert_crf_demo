#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 3:36 PM
# @Author  : sunwenjun
# @File    : tfbert.py
# @brief: PyCharm

import tensorflow as tf
# from bert_encoder import BertEncoder
import logging
from models import crf_lib
from preprocess.ids2tfrecord import SeqLabelDataset
import sys
import time
import copy
import shutil
import logging
import argparse
import collections
from typing import Optional, Tuple, Union
import numpy as np
from transformers.modeling_tf_utils import input_processing, TFModelInputType
# from transformers import TFBertPreTrainedModel, BertConfig, TFBertMainLayer
from transformers import BertConfig, TFBertMainLayer
from models.tf_bert_pretrained_model import TFBertPreTrainedModel

# transformer编码
class TFBert(TFBertPreTrainedModel):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.bert = TFBertMainLayer(config, name="bert")

    def call(
            self,
            input_ids: Optional[TFModelInputType] = None,
            attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
            token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
            position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
            head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
            inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
            training: Optional[bool] = False,
            **kwargs, ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"]
        )

        last_hidden_state = outputs['last_hidden_state']  # batch_size*seq_len*hidden_size
        pooler_output = outputs['pooler_output']  # batch_size*hidden_size

        return last_hidden_state, pooler_output

    def serving_output(self, output):
        return output