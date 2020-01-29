# -*- coding: utf-8 -*-

import logging

from acsa.utils import my_corenlp
from acsa.common import common_path

MODEL_DIR = common_path.common_data_dir + 'stanford-corenlp-full-2018-02-27/'


def create_corenlp_server(start_new_server=False, lang='en'):
    path_or_host = MODEL_DIR
    if not start_new_server:
        path_or_host = 'http://localhost'
    return my_corenlp.StanfordCoreNLP(path_or_host, lang=lang, quiet=False, logging_level=logging.INFO, memory='4g',
                    port=8081)


if __name__ == '__main__':
    create_corenlp_server(start_new_server=True, lang='en')
