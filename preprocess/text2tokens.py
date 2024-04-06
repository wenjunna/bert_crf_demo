#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22
# @Author  : sunwenjun
# @Site    :
# @File    : text2tokens.py


from models.full_tokenizer import FullTokenizer


class Text2Tokens(object):
    def __init__(self, vocab_file="../conf/vocab.txt"):
        '''
        文本转tokens
        Args:
            vocab_file: tokens词典
        '''
        self.tokenizer = FullTokenizer(vocab_file)

    def entity_clean(self, entity_list):
        '''
         entity 预处理 ,脏数据 ,例如下面的例子
        entity_list = 淘宝化妆品销售经理 ["淘宝|COMPANY|0|2","淘宝化妆品销售经理|JOB|0|9","化妆品|JOB_DESC|2|5","销售|JOB|5|7","经理|JOB_LEVEL|7|9"]
        line = "天猫3c数码配件运营	天猫|COMPANY|0|2	天猫3c数码配件运营|JOB|0|10	3c数码配件|INDUSTRY|2|8	运营|JOB|8|10"  # 同一个词处于不同粒度中
        line = "大华,嵌入式	大华|COMPANY|0|2	,|SEPA|2|3	嵌入式|SKILL|3|6	嵌入式|JOB|3|6"  # 同一个次词有多个标签
        Args:
            entity_list:

        Returns:

        '''
        ent_list = []
        for entity in entity_list:
            props = entity.strip().split("|")
            ent = props[0].strip()  # 实体
            tag = props[1].strip()  # 标签
            start = int(props[2].strip())  # 开始下标
            end = int(props[3].strip())  # 结束下标
            if start == end:
                continue
            ent_list.append((ent, tag, start, end))

        # 根据start\end正排
        ent_sorted = sorted(ent_list, key=lambda x: (x[2], x[3]))

        # 去重
        _, _, start_pre, _ = ent_sorted[0]
        ent_sorted_new = [ent_sorted.pop(0)]

        for ent in ent_sorted:
            _, _, start, _ = ent
            if start_pre == start:
                continue
            ent_sorted_new.append(ent)
            start_pre = start

        return ent_sorted_new

    def get_train_samples(self, entity_list):
        '''
        训练数据(验证数据) text转token;训练数据已经有标签
        Args:
            entity_list: ['招聘|STOP|0|2', '普工|JOB|2|4', '月薪5000|SALARY|4|10', '包食宿|JOB_PROP|10|13']

        Returns:
            招	B_STOP
            聘	I_STOP
            普	B_JOB
            工	I_JOB
            月	B_SALARY
            薪	I_SALARY
            5000	I_SALARY
            包	B_JOB_PROP
            食	I_JOB_PROP
            宿	I_JOB_PROP

        '''
        ent_list = self.entity_clean(entity_list)  # 预处理

        token_list = []
        tag_list = []
        for entity in ent_list:
            (ent, tag, start, end) = entity
            tokens = self.tokenizer.tokenize(ent)  # 以实体为单位
            if len(tokens) < 1:
                continue

            token_list += tokens

            begin_tag = "B_" + tag
            inner_tag = "I_" + tag
            tag_list.append(begin_tag)
            for _ in tokens[1:]:
                tag_list.append(inner_tag)

        samples = zip(token_list, tag_list)

        return samples

    def get_test_samples(self, text):
        '''
        测试数据,text转token;tag初始标签默认为o,id默认为0
        Args:
            text: 招聘普工月薪5000包食宿

        Returns:
            招	O
            聘	O
            普	O
            工	O
            月	O
            薪	O
            5000	O
            包	O
            食	O
            宿	O
        '''

        token_list = self.tokenizer.tokenize(text)
        tag_list = ["O"] * len(token_list)
        samples = zip(token_list, tag_list)

        return samples


if __name__ == '__main__':
    line = "招聘普工月薪5000包食宿	招聘|STOP|0|2	普工|JOB|2|4	月薪5000|SALARY|4|10	包食宿|JOB_PROP|10|13"
    items = line.strip().split("\t")
    text = items[0]
    ent_list = items[1:]
    print("text:", text)
    print("ent_list", ent_list)
    pro = Text2Tokens(vocab_file="../conf/vocab.txt")

    print("-----测试数据-----")
    samples = pro.get_test_samples(text)
    for token, tag in samples:
        print("%s\t%s" % (token, tag))

    print("-----训练数据-----")
    samples = pro.get_train_samples(ent_list)
    for token, tag in samples:
        print("%s\t%s" % (token, tag))
