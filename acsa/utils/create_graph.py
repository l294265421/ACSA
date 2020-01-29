# -*- coding: utf-8 -*-

"""
1. 基于句法树构建图
2. 基于共指关系构建图
"""

import numpy as np
import spacy
import networkx as nx
from spacy import displacy
import dgl
import matplotlib.pyplot as plt
import torch
# import neuralcoref

from acsa.utils import corenlp_factory
from acsa.utils import word_processor
from acsa.utils import tokenizers

spacy_dependencies = ['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux',
                      'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative',
                      'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj',
                      'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet',
                      'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp', 'next', 'coref', 'self_loop',
                      'subtok']


def create_dependency_graph(sentence: str, stanford_nlp):
    arcs, words = stanford_nlp.dependency_parse(sentence, True)
    word_num = len(words)
    word_relation_graph = np.zeros((word_num, word_num))
    for i in range(word_num):
        word_relation_graph[i][i] = 1
    for word_relation in arcs[1:]:
        head = word_relation[2] - 1
        dependency = word_relation[1] - 1
        word_relation_graph[head][dependency] = 1
        word_relation_graph[dependency][head] = 1
    return word_relation_graph


def get_coref_edges(text: str, stanford_nlp):
    all_words = []
    edges = []
    if stanford_nlp is None:
        return edges, all_words
    coref_relations, sentence_words = stanford_nlp.coref(text, return_words=True)
    sentence_start_indices = []
    word_num = 0
    for i in range(len(sentence_words)):
        sentence_word = sentence_words[i]
        sentence_start_indices.append(word_num)
        word_num += len(sentence_word)
        for word in sentence_word:
            all_words.append(word['word'])
    for coref_relation in coref_relations:
        for i in range(len(coref_relation)):
            for j in range(i + 1, len(coref_relation)):
                first_word = coref_relation[i]
                second_word = coref_relation[j]
                for k in range(first_word[1], first_word[2]):
                    first_index = sentence_start_indices[first_word[0] - 1] + k - 1
                    for l in range(second_word[1], second_word[2]):
                        second_index = sentence_start_indices[second_word[0] - 1] + l -1
                        edges.append([first_index, second_index, 'coref'])
                        edges.append([second_index, first_index, 'coref'])
    return edges, all_words


def create_dependency_graph_by_spacy(sentence: str, spacy_nlp):
    # https://spacy.io/docs/usage/processing-text
    document = spacy_nlp(sentence)
    seq_len = len([token for token in document])
    matrix = np.zeros((seq_len, seq_len)).astype('float32')

    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
    return matrix


def create_dependency_graph_for_dgl(sentence: str, spacy_nlp, stanford_nlp=None):
    # https://spacy.io/docs/usage/processing-text
    document = spacy_nlp(sentence)
    seq_len = len([token for token in document])
    g = dgl.DGLGraph()
    g.add_nodes(seq_len)
    edge_list = []
    for token in document:
        if token.i < seq_len:
            edge_list.append((token.i, token.i, 'self_loop'))
            if token.i + 1 < seq_len:
                edge_list.append((token.i, token.i + 1, 'next'))
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    if not child.dep_:
                        continue
                    edge_list.append((token.i, child.i, child.dep_))
                    edge_list.append((child.i, token.i, child.dep_))

    coref_edges, coref_words = get_coref_edges(sentence, stanford_nlp)
    if len(coref_words) == seq_len and len(coref_edges) != 0:
        edge_list.extend(coref_edges)
    src, dst, rtype = list(zip(*edge_list))
    rtype_index = [spacy_dependencies.index(r) for r in rtype]
    g.add_edges(src, dst)
    g.edata.update({'rel_type': torch.tensor(rtype_index)})
    return g


def create_aspect_term_dependency_graph(aspect_term_indices, polarity_indices, words):
    connective_and_relation_pair = {
        'other than': '',  # Food other than sushi is also very nice.

    }
    all_dependencies = ['self-loop', 'inter']
    g = dgl.DGLGraph()
    g.add_nodes(len(words))
    edge_list = []
    for k in range(len(words)):
        edge_list.append((k, k, 'self-loop'))
    for k in range(len(aspect_term_indices)):
        polarity_index = polarity_indices[k]
        aspect_term_index = aspect_term_indices[k]
        if polarity_index == -100:
            continue
        if k + 1 < len(aspect_term_indices) and polarity_indices[k + 1] != -100:
            for i in range(aspect_term_index[0], aspect_term_index[1] + 1):
                for j in range(aspect_term_indices[k + 1][0], aspect_term_indices[k + 1][1] + 1):
                    edge_list.append((i, j, 'inter'))
                    edge_list.append((j, i, 'inter'))
    src, dst, rtype = list(zip(*edge_list))
    rtype_index = [all_dependencies.index(r) for r in rtype]
    g.add_edges(src, dst)
    g.edata.update({'rel_type': torch.tensor(rtype_index)})
    return g






def plot_dgl_graph(graph):
    nx_G = graph.to_networkx()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()


def create_coref_graph(text: str, stanford_nlp):
    coref_relations, sentence_words = stanford_nlp.coref(text, return_words=True)
    all_words = []
    sentence_start_indices = []
    word_num = 0
    for i in range(len(sentence_words)):
        sentence_word = sentence_words[i]
        sentence_start_indices.append(word_num)
        word_num += len(sentence_word)
        for word in sentence_word:
            all_words.append(word['word'])
    word_relation_graph = np.zeros((word_num, word_num))
    for coref_relation in coref_relations:
        for i in range(len(coref_relation)):
            for j in range(i + 1, len(coref_relation)):
                first_word = coref_relation[i]
                second_word = coref_relation[j]
                for k in range(first_word[1], first_word[2]):
                    first_index = sentence_start_indices[first_word[0] - 1] + k - 1
                    for l in range(second_word[1], second_word[2]):
                        second_index = sentence_start_indices[second_word[0] - 1] + l -1
                        print('first_word: %s second_word: %s' % (all_words[first_index], all_words[second_index]))
                        word_relation_graph[first_index][second_index] = 1
                        word_relation_graph[second_index][first_index] = 1
    return word_relation_graph


if __name__ == '__main__':
    sentence = 'When the food came, it was almost good.'
    core_nlp = corenlp_factory.create_corenlp_server()
    coref_graph = get_coref_edges(sentence, core_nlp)

    spacy_nlp = spacy.load("en_core_web_sm")
    doc = spacy_nlp(sentence)
    for token in doc:
        children = list(token.children)
        print(token)
    # neuralcoref.add_to_pipe(spacy_nlp)
    graph = create_dependency_graph_for_dgl(sentence, spacy_nlp)
    plot_dgl_graph(graph)
    print('')
