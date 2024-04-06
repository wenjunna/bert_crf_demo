#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22
# @Author  : sunwenjun
# @Site    :
# @File    : model_test.py

import argparse
import logging

import numpy as np
import tensorflow as tf

import crf_lib
# import nlu_dataset
import to_tfrecord
# from nlu_model import HiGRUCRFModel
from bert_crf import BertCRFModel
from full_tokenizer import FullTokenizer
import logging
from transformers import BertConfig, TFBertMainLayer
from tf_bert_pretrained_model import TFBertPreTrainedModel

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def id_to_tag(file="./conf/label2tag_20210917.txt"):
    id2tag = {}
    for line in open(file, 'r'):
        items = line.strip().split('\t')
        if len(items) != 2:
            print(items)
            continue

        id = int(items[0])
        tag = items[1]

        id2tag[id] = tag

    return id2tag


# 输出可读形式
# token -> text
# tags -> k可读


def format_res(text_list, pred_list):
    # real_text = "".join(text_list)
    # 一个token可能由多个字符组成
    index = 0
    start = 0
    start_ch = 0
    end_ch = 0
    length_ch = 0
    length = 0
    end = start + length
    res_list = ["".join(text_list)]
    tagger = ""
    for token, tag in zip(text_list, pred_list):
        if index == 0 and tag.startswith("B_"):
            tagger = tag.strip().split("_", 1)[1]

        if index > 0 and tag.startswith("B_"):
            if start < end:  # 保存上一个
                word = "".join(text_list[start:end])
                res_list.append(word + "|" + tagger + "|" + str(start_ch) + "|" + str(end_ch))
            tagger = tag.strip().split("_", 1)[1]  # 开始新的
            start = end
            start_ch = end_ch
            length = 0
            length_ch = 0

        length_ch += len(token)
        end_ch = start_ch + length_ch
        length += 1
        end = start + length  # start end 用于控制list下标； start_ch,end_ch 用于控制 字符下表
        index += 1

    if start < end:
        word = "".join(text_list[start:end])
        res_list.append(word + "|" + tagger + "|" + str(start_ch) + "|" + str(end_ch))
    return res_list


def test_model(model, test_ds, vocab_file, tag2id_file, test_res):
    tokenizer = FullTokenizer(vocab_file)
    id2tag = id_to_tag(tag2id_file)
    test_res = open(test_res, 'w', encoding='utf8')

    it = iter(test_ds)
    total_cnt = 0

    try:
        while True:
            data, _ = next(it)  # 一个batch size
            # text = data['text']
            word_ids = data['word_ids']
            mask_ids = data['mask_ids']
            type_ids = data['type_ids']
            # label = data['label']
            text = [word_ids, mask_ids, type_ids]

            bert_ebd_outputs, pred = model(text)
            for t_one, p_one in zip(word_ids, pred):
                t_list = list(t_one.numpy())
                p_list = list(p_one.numpy())
                total_cnt += 1

                t_tokens = tokenizer.ids2tokens(t_list)
                p_tags = [id2tag.get(id_, "O") for id_ in p_list]

                text_list = []
                pred_list = []
                for t1, p1 in zip(t_tokens, p_tags):
                    if t1 in ['[CLS]', '[SEP]']:  # 101 cls, 102 sep
                        continue
                    if t1 == '[PAD]':
                        break
                    # test_res.write("%s\t%s\n" % (t1, p1))
                    text_list.append(t1)
                    pred_list.append(p1)

                # 转成可读格式
                res_list = format_res(text_list, pred_list)
                test_res.write("\t".join(res_list) + "\n")
                # test_res.write("\n")
    except:
        print("done")
    logging.info("total cnt %d" % total_cnt)
    return True


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vocab_file", type=str, default="./conf/vocab.txt", help="vocab_file")
    parser.add_argument("--label2tag_file", type=str, default="./conf/label2tag_20210917.txt", help="label2tag_file")
    parser.add_argument("--test_file", type=str, default="./data/test.tfrecord", help="Test files path")
    parser.add_argument("--test_res_file", type=str, default="./data/test_res.txt", help="Test res files path")
    parser.add_argument("--weight_dir", type=str, default="weights/HigruCrfModel/HigruCrfModel",
                        help="saving weight path")
    parser.add_argument("--model_dir", type=str, default="./models/higru_crf", help="saving model path")
    parser.add_argument("--n_vocab", type=int, default="21862", help="n_vocab")
    parser.add_argument("--n_label", type=int, default="65", help="n_label")
    parser.add_argument("--seq_length", type=int, default="512", help="seq_length")
    parser.add_argument("--batch_size", type=int, default="64", help="batch size")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    data_set = to_tfrecord.SeqLabelDataset(seq_length=args.seq_length)
    test_ds = data_set.load_tfrecord(args.test_file, args.batch_size)

    bert_config = BertConfig.from_pretrained("bert-base-chinese", cache_dir="./bert_pretrain")
    bert_config.vocab_size = args.n_vocab
    bert_config.hidden_dropout_prob = 0.0
    bert_config.attention_probs_dropout_prob = 0.0
    bert_config.classifier_dropout = 0.0
    bert_crf_model2 = BertCRFModel(
        bert_config,
        n_label=args.n_label,
        drop_rate=0.0)
    model_dev = bert_crf_model2.build(args.seq_length)

    model_dev.load_weights(args.weight_dir)

    pred = test_model(model_dev, test_ds, args.vocab_file, args.tag2id_file, args.test_res_file)
    print("done")

# test_one()
