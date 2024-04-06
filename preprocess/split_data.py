#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/31 7:54 PM
# @Author  : sunwenjun
# @File    : split_data.py
# @brief: 分训练集,验证集,测试集

import random


def split_train_dev_test(in_file, train_file, dev_file, test_file):
    '''
    分训练集\验证集\测试集,并保存txt文件
    :param in_file:输入文件，txt格式，一行一个数据
    :param train_file: 训练集
    :param dev_file: 验证集
    :param test_file: 测试集
    :return:
    '''
    train = open(train_file, 'w', encoding='utf8')
    dev = open(dev_file, 'w', encoding='utf8')
    test = open(test_file, 'w', encoding='utf8')
    train_cnt, dev_cnt, test_cnt = 0, 0, 0

    for line in open(in_file, 'r', encoding='utf8'):
        rnd = random.random()  # 产生（0，1）之间的随机数
        if rnd < 0.8:
            train.write(line)
            train_cnt += 1
        elif rnd < 0.9:
            dev.write(line)
            dev_cnt += 1
        else:
            test.write(line)
            test_cnt += 1
    return train_cnt, dev_cnt, test_cnt


if __name__ == '__main__':
    # 分训练集\验证集\测试集
    in_file = "../testdata/data.txt"
    train_file = "../testdata/train.txt"
    dev_file = "../testdata/dev.txt"
    test_file = "../testdata/test.txt"

    train_cnt, dev_cnt, test_cnt = split_train_dev_test(in_file=in_file, train_file=train_file,
                                                        dev_file=dev_file, test_file=test_file)
    print("train_cnt: ", train_cnt)
    print("dev_cnt: ", dev_cnt)
    print("test_cnt: ", test_cnt)
