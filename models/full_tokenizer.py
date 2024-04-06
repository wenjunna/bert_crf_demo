#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
#
########################################################################

"""
File: tokenization.py
Author:
Date: 2019/11/06 11:53:24
Brief: 在bert的 toknization 上进行修改
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import re
import sys
import six
import logging
import unicodedata
import collections

import math
import numpy as np


# 字符类型
class CharType(object):
    @staticmethod
    def is_whitespace(char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    @staticmethod
    def is_control(char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat in ("Cc", "Cf"):
            return True
        return False

    @staticmethod
    def is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
        return False

    @staticmethod
    def is_punctuation(char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False


# 中文覆盖率统计
def chinese_overlap_ratio(text):
    ch_num = 0
    all_num = 0
    for ch in text:
        if CharType.is_chinese_char(ord(ch)):
            ch_num += 1
        all_num += 1
    return ch_num / (all_num + 0.001)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""
    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def to_str(string, encoding="utf-8"):
    """convert to str for print"""
    if isinstance(string, bytes):
        return string.decode(encoding)
    return string


# 文本预处理
def preprocess_text(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())
    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()
    return outputs


class BasicTokenizer(object):
    """
        Runs basic tokenization (punctuation splitting, lower casing, etc.)
    """

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.char_type = CharType()

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # 中文前后加空格
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            punc = self._run_split_on_punc(token)
            split_tokens.extend(punc)
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def text_norm(self, text):
        """
        basic norm text add by wy
        """
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        tok_list = []
        for tok in orig_tokens:
            tok = self._run_strip_accents(tok)
            tok_list.append(tok)
        return " ".join(tok_list)

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        output = []
        start_new_word = True
        while i < len(chars):
            char = chars[i]
            if self.char_type.is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        ## 中文CJK 前后加空格
        output = []
        for char in text:
            cp = ord(char)
            if self.char_type.is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _clean_text(self, text):
        # 删除控制字符 + 归一化空白字符
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self.char_type.is_control(char):
                continue
            if self.char_type.is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """
        Runs WordPiece tokenziation.
    """

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
        input = "unaffable"
        output = ["un", "##aff", "##able"]

        Args:
        text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
        A list of wordpiece tokens.
        """
        text = convert_to_unicode(text)
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start: end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file,
                 do_lower_case=True,
                 spm_model_file=None,
                 do_sub_tok=False):
        '''
        Args:
            vocab_file: 词典
            do_lower_case: 转小写
            do_sub_tok: 片段拆分
            sp_model_file: sentencepiece 模型文件
        '''
        self.do_lower = do_lower_case
        self.do_sub_tok = do_sub_tok

        self.vocab = None
        self.sp_model = None
        if spm_model_file:
            logging.info("#Use spm_model_file")

            import sentencepiece as spm
            self.sp_model = spm.SentencePieceProcessor()
            logging.info("loading sentence piece model")
            self.sp_model.Load(spm_model_file)
            self.vocab = {self.sp_model.IdToPiece(i): i for i
                          in range(self.sp_model.GetPieceSize())}
        else:
            logging.info("#Use vocab_file")
            self.vocab = self.load_vocab(vocab_file)
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

        self.char_type = CharType()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        if self.sp_model:
            split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
            return split_tokens

        # vocab token
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            # print("token-->", token)
            if self.do_sub_tok:
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
            else:
                split_tokens.append(token)
        return split_tokens

    # tok -> ids
    def tokens2ids(self, tokens):
        if self.sp_model:
            return [self.sp_model.PieceToId(printable_text(tok)) for tok in tokens]
        else:
            unk_id = self.vocab["[UNK]"]
            return [self.vocab.get(tok, unk_id) for tok in tokens]

    # id -> tok
    def ids2tokens(self, ids):
        if self.sp_model:
            return [self.sp_model.IdToPiece(id_) for id_ in ids]
        else:
            # res = []
            # for id_ in ids:
            #     res.append(self.inv_vocab[id_])
            return [self.inv_vocab.get(id_, '[PAD]') for id_ in ids]

    def get_vocab_words(self):
        return self.vocab

    def load_vocab(self, vocab_file):
        vocab = collections.OrderedDict()
        with open(vocab_file, "r",encoding='utf8') as infile:
            for line in infile:
                line = convert_to_unicode(line)
                if not line:
                    break
                line = line.strip()
                if line not in vocab:
                    vocab[line] = len(vocab)
        return vocab


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    """turn sentences into word pieces."""
    SPIECE_UNDERLINE = u"▁".encode("utf-8")
    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []

    for piece in pieces:
        piece = printable_text(piece)
        if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))

            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)
    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids
