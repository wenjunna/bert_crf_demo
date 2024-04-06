#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 11:14 AM
# @Author  : sunwenjun
# @File    : preprocess_main.py
# @brief: 训练数据预处理

from preprocess.split_data import split_train_dev_test
from preprocess.text2tokens import Text2Tokens
from preprocess.tokens2ids import Tokens2Ids
from preprocess.ids2tfrecord import SeqLabelDataset


def line2tokens(infile, outfile, withpv, sign):
    '''
    文本转tokens
    Args:
        in_file:输入的训练数据文件  pv \t text \t entity1 \t entity2
        out_file: 输出token，包括文本token和标签token   token \t tag
        with_pv: 第一列是否是pv列，训练数据里没有pv列（针对我的数据）
        sign: 数据标志 train/dev/test

    Returns:

    '''
    out_tokens = open(outfile, 'w', encoding='utf8')
    text2tokes = Text2Tokens(vocab_file="../conf/vocab.txt")

    for line in open(infile, 'r', encoding='utf8'):
        items = line.strip().split("\t")
        if len(items) < 3:
            continue

        if withpv:
            pv = items.pop(0)
            pv = int(pv)
            if pv < 100:
                break

        text = items.pop(0)
        ent_list = items
        samples = []
        if sign == "train" or sign == "dev":
            samples = text2tokes.get_train_samples(ent_list)
        elif sign == "test":
            samples = text2tokes.get_test_samples(text)
        # save
        tokens = []
        for token, tag in samples:
            tokens.append(token)
            out_tokens.write("%s\t%s\n" % (token, tag))
        out_tokens.write("\n")

        # 打印badcase  token连接后的长度和text长度不一致，前期可以忽略，后期人工标注
        if len("".join(tokens)) != len("".join(text.strip().split())):
            print("badcase: %s " % line)

    return True


def process(data_file):
    withpv = False
    vocab_file = "../conf/vocab.txt"
    label2tag_file = "../conf/label2tag_20210917.txt"

    # 第一步，split data  分训练集\验证集\测试集
    train_file = "../testdata/train.txt"
    dev_file = "../testdata/dev.txt"
    test_file = "../testdata/test.txt"
    train_cnt, dev_cnt, test_cnt = split_train_dev_test(in_file=data_file, train_file=train_file,
                                                        dev_file=dev_file, test_file=test_file)
    print("train_cnt: ", train_cnt)
    print("dev_cnt: ", dev_cnt)
    print("test_cnt: ", test_cnt)

    # 第二步，ext - > tokens
    train_tokens_file = "../testdata/train_tokens.txt"
    dev_tokens_file = "../testdata/dev_tokens.txt"
    test_tokens_file = "../testdata/test_tokens.txt"
    line2tokens(infile=train_file, outfile=train_tokens_file, withpv=withpv, sign="train")
    line2tokens(infile=dev_file, outfile=dev_tokens_file, withpv=withpv, sign="dev")
    line2tokens(infile=test_file, outfile=test_tokens_file, withpv=withpv, sign="test")

    # 第三步，tokens - > ids
    train_ids_file = '../testdata/train_ids.txt'
    dev_ids_file = "../testdata/dev_ids.txt"
    test_ids_file = "../testdata/test_ids.txt"
    data_set = Tokens2Ids(vocab_file=vocab_file, label_file=label2tag_file)
    data_set.convert2id(infile=train_tokens_file, outfile=train_ids_file)
    data_set.convert2id(infile=dev_tokens_file, outfile=dev_ids_file)
    data_set.convert2id(infile=test_tokens_file, outfile=test_ids_file)
    print("done")

    # 第四步，ids -> tfrecord
    seq_length = 512  # 注意bert为512
    dataset = SeqLabelDataset(seq_length=seq_length)
    train_tfrecord = "../testdata/train.tfrecord"  # 输出的tfrecord格式文件
    dev_tfrecord = "../testdata/dev.tfrecord"  # 输出的tfrecord格式文件
    test_tfrecord = "../testdata/test.tfrecord"  # 输出的tfrecord格式文件
    dataset.convert2tfrecord(infile=train_ids_file, outfile=train_tfrecord)
    dataset.convert2tfrecord(infile=dev_ids_file, outfile=dev_tfrecord)
    dataset.convert2tfrecord(infile=test_ids_file, outfile=test_tfrecord)

    # dataset.load_tfrecord(file=outfile)

    return


if __name__ == '__main__':
    data_file = "../testdata/data.txt"
    process(data_file)
    print("done")
