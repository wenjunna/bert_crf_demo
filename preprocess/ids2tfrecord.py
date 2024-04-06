#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22
# @Author  : sunwenjun
# @Site    :
# @File    : ids2tfrecord.py

import tensorflow as tf


class SeqLabelDataset(object):
    '''序列标注 dataset'''

    def __init__(self, seq_length=32):
        self.seq_length = seq_length

    def fixed_length(self, x, pad=0):
        '''
        固定输长度
        Args:
            x:
            pad:

        Returns:

        '''
        while len(x) < self.seq_length:
            x.append(pad)
        return x[:self.seq_length]

    def load_id_file(self, file):
        # x由一个部分组成，token_ids
        xs = []
        ys = []
        x = []
        y = []
        for line in open(file, 'r'):
            items = line.strip().split('\t')
            if len(items) != 2:
                x = self.fixed_length(x, pad=0)
                y = self.fixed_length(y, pad=0)
                assert len(x) == self.seq_length
                assert len(y) == self.seq_length

                ys.append(y)
                xs.append(x)
                x = []
                y = []
                continue
            x.append(int(items[0]))
            y.append(int(items[1]))
        return xs, ys

    def load_id_file_v2(self, file):
        # x由三个部分组成，token_ids\mask_ids\type_ids
        xs = []
        ys = []
        x = []
        mask_ids = []
        type_ids = []
        y = []
        for line in open(file, 'r'):
            items = line.strip().split('\t')
            if len(items) != 2:
                x = self.fixed_length(x, pad=0)
                mask_ids = self.fixed_length(mask_ids, pad=0)
                type_ids = self.fixed_length(type_ids, pad=0)
                y = self.fixed_length(y, pad=0)
                assert len(x) == self.seq_length
                assert len(y) == self.seq_length
                assert len(mask_ids) == self.seq_length
                assert len(type_ids) == self.seq_length

                ys.append(y)
                xs.append((x, mask_ids, type_ids))

                x = []
                mask_ids = []
                type_ids = []
                y = []
                continue
            x.append(int(items[0]))
            mask_ids.append(1)
            type_ids.append(0)
            y.append(int(items[1]))
        return xs, ys

    def convert2tfrecord(self, infile, outfile):
        '''
        ids -> tfrecord
        Args:
            fileIn: 输入ids文件
            fileOut: tfrecord文件

        Returns:

        '''
        # xs, ys = self.load_id_file(fileIn)
        xs, ys = self.load_id_file_v2(infile)
        local_writer = tf.io.TFRecordWriter(outfile)

        for (word_ids, mask_ids, type_ids), label in zip(xs, ys):
            example = tf.train.Example(features=tf.train.Features(
                # feature={
                #     "text": tf.train.Feature(int64_list=tf.train.Int64List(value=text)),
                #     "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                # }
                feature={
                    "word_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=word_ids)),
                    "mask_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=mask_ids)),
                    "type_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=type_ids)),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                }
            )
            )
            local_writer.write(example.SerializeToString())
        local_writer.close()
        return True

    def load_tfrecord(self, file, batch_size=32):
        '''
        加载tfrecord格式文件
        Args:
            file:
            batch_size:

        Returns:

        '''
        # feature_description = {
        #     'text': tf.io.FixedLenFeature([self.seq_length], tf.int64),
        #     'label': tf.io.FixedLenFeature([self.seq_length], tf.int64),
        # }
        feature_description = {
            'word_ids': tf.io.FixedLenFeature([self.seq_length], tf.int64),
            'mask_ids': tf.io.FixedLenFeature([self.seq_length], tf.int64),
            'type_ids': tf.io.FixedLenFeature([self.seq_length], tf.int64),
            'label': tf.io.FixedLenFeature([self.seq_length], tf.int64),
        }

        def _parse_int32(exam_proto):  # 映射函数，用于解析一条example
            example = tf.io.parse_single_example(exam_proto, feature_description)
            parsed = {}
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t, tf.int32)
                parsed[name] = t
            return parsed, parsed['label']

        def _parse_function(exam_proto):  # 映射函数，用于解析一条example
            parsed = tf.io.parse_single_example(exam_proto, feature_description)
            return parsed, parsed['label']

        dataset = tf.data.TFRecordDataset(file)
        # dataset = dataset.map(_parse_function)
        dataset = dataset.map(_parse_int32)
        dataset = dataset.batch(batch_size)
        return dataset


if __name__ == '__main__':
    seq_length = 512  # 注意bert为512
    vocab_file = "../conf/vocab.txt"
    label2tag_file = "../conf/label2tag_20210917.txt"
    dataset = SeqLabelDataset(seq_length=seq_length)

    infile = "../testdata/train_ids.txt"  # ids文件
    outfile = "../testdata/train.tfrecord"  # 输出的tfrecord格式文件
    dataset.convert2tfrecord(infile=infile, outfile=outfile)

    infile = "../testdata/dev_ids.txt"  # ids文件
    outfile = "../testdata/dev.tfrecord"  # 输出的tfrecord格式文件
    dataset.convert2tfrecord(infile=infile, outfile=outfile)

    infile = "../testdata/test_ids.txt"  # ids文件
    outfile = "../testdata/test.tfrecord"  # 输出的tfrecord格式文件
    dataset.convert2tfrecord(infile=infile, outfile=outfile)

    # dataset.load_tfrecord(file=outfile)
    print("done")
