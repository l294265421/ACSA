# -*- coding: utf-8 -*-
"""

具有统一输入输出的不同分词器:
输入：text
输出：list of word
"""

import re

import jieba
import spacy
from nltk import tokenize

from acsa.utils import word_processor
from acsa.utils import corenlp_factory


class BaseTokenizer:
    """
    分词器基类
    """

    def __init__(self, word_processor=word_processor.BaseWordProcessor()):
        self.word_processor = word_processor

    def is_valid_text(self, text):
        if text is None:
            return False
        return True

    def _inner_segment(self, text):
        """

        :param text:
        :return:
        """
        words = text.split(' ')
        return words

    def _segment(self, text):
        result = []
        words = self._inner_segment(text)
        for word in words:
            word = self.word_processor.process(word)
            if word is not None:
                result.append(word)
        return result

    def __call__(self, text: str) -> list:
        if not self.is_valid_text(text):
            return []
        else:
            words = self._segment(text)
            result = []
            for word in words:
                word_temp = word
                while True:
                    hyphen_index = word_temp.find('-')
                    if hyphen_index != -1:
                        if hyphen_index != 0:
                            result.append(word_temp[: hyphen_index])
                        result.append('-')
                        word_temp = word_temp[hyphen_index + 1:]
                    else:
                        if word_temp != '':
                            result.append(word_temp)
                        break
            return result


class JiebaTokenizer(BaseTokenizer):
    """
    jieba分词器
    """

    def __init__(self, word_processor=word_processor.BaseWordProcessor()):
        super().__init__(word_processor)

    def _inner_segment(self, text):
        if not self.is_valid_text(text):
            return []
        return list(jieba.cut(text))


class NltkTokenizer(BaseTokenizer):
    """
    jieba分词器
    """

    def __init__(self, word_processor=word_processor.BaseWordProcessor()):
        super().__init__(word_processor)

    def _inner_segment(self, text):
        if not self.is_valid_text(text):
            return []
        return tokenize.word_tokenize(text)


class StanfordTokenizer(BaseTokenizer):
    """
    jieba分词器
    """

    def __init__(self, word_processor=word_processor.BaseWordProcessor(), lang='en'):
        super().__init__(word_processor)
        self.stanford_nlp = corenlp_factory.create_corenlp_server(lang=lang)

    def _inner_segment(self, text):
        if not self.is_valid_text(text):
            return []
        words = self.stanford_nlp.word_tokenize(text)
        return words


class SpacyTokenizer(BaseTokenizer):
    """
    jieba分词器
    """

    def __init__(self, word_processor=word_processor.BaseWordProcessor()):
        super().__init__(word_processor)
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def _inner_segment(self, text):
        if not self.is_valid_text(text):
            return []
        doc = self.spacy_nlp(text, disable=["parser"])
        words = [token.text for token in doc]
        return words


if __name__ == '__main__':
    text = 'Food-awesome.'
    tokenizer = SpacyTokenizer()
    words = tokenizer(text)
    print(words)

