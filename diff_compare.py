#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22
# @Author  : sunwenjun
# @Site    :
# @File    : diff_compare.py

def diff_compare(old_file, new_file, diff_file, same_file):
    '''
    对比以前标注结果和bert_crf模型给出结果的异同
    :param old_file: text  \t  ent1  \t  ent2
    :param new_file: text  \t  ent1  \t  ent2
    :param diff_file:
    :param same_file:
    :return:
    '''
    new_list = []
    old_list = []
    # 读取新标注
    for line in open(new_file, 'r', encoding='utf8'):
        items = line.strip().split("\t")
        new_list.append(items)

    total_cnt = len(new_list)

    # 读取老标注
    for line in open(old_file, 'r', encoding='utf8'):
        tmp = []
        items = line.strip().split("\t")
        # pv = items.pop(0)
        text = items.pop(0)
        tmp.append(text)
        for item in items:
            entity = item.strip().split("|")
            tmp.append("|".join(entity[:-1]))

        old_list.append(tmp)
        if len(old_list) == total_cnt:
            break

    # 对比，save
    diff_cnt, same_cnt, total_cnt = 0, 0, 0
    diff = open(diff_file, 'w', encoding='utf8')
    same = open(same_file, 'w', encoding='utf8')
    for new, old in zip(new_list, old_list):
        total_cnt += 1
        if new != old:
            diff_cnt += 1
            diff.write("%s\t%s\n" % ("new", "\t".join(new)))
            diff.write("%s\t%s\n" % ("old", "\t".join(old)))
            diff.write("\n")
            continue
        same.write("%s\n" % "\t".join(old))
        same_cnt += 1

    return total_cnt, same_cnt, diff_cnt


if __name__ == '__main__':
    old_file = "../data/test.txt"
    new_file = "./data/test_res.txt"
    diff_file = "./data/diff.txt"
    same_file = "./data/same.txt"

    total_cnt, same_cnt, diff_cnt = diff_compare(old_file, new_file, diff_file, same_file)
    print("done", total_cnt, same_cnt, diff_cnt)
