#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 10:34 AM
# @Author  : sunwenjun
# @File    : tokens2ids.py
# @brief: PyCharm

class Tokens2Ids(object):
    def __init__(self, vocab_file="../conf/vocab.txt", label_file="../conf/label2tag.txt"):
        '''
        token转换成对应的id
        Args:
            vocab_file:
            label_file:
        '''
        self.start = '[CLS]'
        self.end = '[SEP]'
        self.unk = '[UNK]'
        self.pad = '[PAD]'
        self.tagO = 'O'
        self.vocab_file = vocab_file
        self.label_file = label_file
        self.vocab_dic = self.load_vocab(self.vocab_file)
        self.label_dic = self.load_label(self.label_file)

    def get_vocab(self):
        return self.vocab_dic

    def get_label(self):
        return self.label_dic

    def load_vocab(self, file_name):
        '''
        加载词典
        Args:
            file_name: 词典文件

        Returns:

        '''
        kv_dic = {}
        print("load vocab dict=" + file_name)
        id = 0
        for line in open(file_name):
            kv_dic[line.strip()] = id
            id += 1
        vocab_dic = kv_dic
        return vocab_dic

    def load_label(self, file_name):
        '''
        加载标签
        Args:
            file_name: 标签文件

        Returns:

        '''
        kv_dict = {}
        print("load label dict=" + file_name)
        for line in open(file_name):
            if line.startswith('##'):
                continue
            items = line.strip().split('\t')
            if len(items) != 2:
                continue
            id = int(items[0])
            label = items[1]
            kv_dict[label] = id
        label_dic = kv_dict
        return label_dic

    def sample_ids(self, sample_tokens, sample_labels):
        '''
        token -> id, tag -> id
        Args:
            sample_tokens: []
            sample_labels: []

        Returns:

        '''
        unk_id = self.vocab_dic[self.unk]
        tag_O_id = self.label_dic[self.tagO]
        token_ids = [self.vocab_dic.get(self.start, unk_id)]
        label_ids = [tag_O_id]
        for token, label in zip(sample_tokens, sample_labels):
            token_ids.append(self.vocab_dic.get(token, unk_id))
            label_ids.append(self.label_dic.get(label, tag_O_id))

        token_ids.append(self.vocab_dic.get(self.end, unk_id))
        label_ids.append(tag_O_id)
        return [token_ids, label_ids]

    def convert2id(self, infile='../testdata/train_tokens.txt', outfile='../testdata/train_ids.txt'):
        '''
        token -> id, tag -> id
        Args:
            infile:
            outfile:

        Returns:

        '''
        samples = []
        sample_tokens = []
        sample_labels = []

        for line in open(infile, 'r'):
            items = line.strip().split('\t')
            if len(items) != 2:  # 两个字符串分割部分
                sample = self.sample_ids(sample_tokens, sample_labels)
                samples.append(sample)
                sample_tokens = []
                sample_labels = []
                continue

            sample_tokens.append(items[0])
            sample_labels.append(items[1])

        fout = open(outfile, 'w')
        for token_ids, label_ids in samples:

            for token_id, label_id in zip(token_ids, label_ids):
                fout.write('{}\t{}\n'.format(token_id, label_id))

            fout.write('\n')
        fout.close()
        return


if __name__ == '__main__':
    vocab_file = "../conf/vocab.txt"
    label2tag_file = "../conf/label2tag_20210917.txt"
    input_file = '../testdata/train_tokens.txt'
    output_file = '../testdata/train_ids.txt'

    data_set = Tokens2Ids(vocab_file=vocab_file, label_file=label2tag_file)
    data_set.convert2id(infile=input_file, outfile=output_file)
    print("done")
