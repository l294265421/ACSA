from typing import *
import copy

import numpy as np
import torch
import torch.nn as nn
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.training import metrics
from allennlp.nn import util as allennlp_util
import dgl
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from allennlp.nn import util as nn_util

from acsa.utils import attention_visualizer


class AttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze()
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class DotProductAttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.uw = nn.Linear(in_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        similarities = self.uw(h)
        similarities = similarities.squeeze()
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class AverageAttention(nn.Module):
    """
    2019-emnlp-Attention is not not Explanation
    """

    def __init__(self):
        super().__init__()

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        alpha = allennlp_util.masked_softmax(mask, mask)
        return alpha


class BernoulliAttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze()
        alpha = torch.sigmoid(similarities)
        return alpha


class AttentionInCan(nn.Module):
    """
    2019-emnlp-CAN Constrained Attention Networks for Multi-Aspect Sentiment Analysis
    """

    def __init__(self, in_features, bias=True, softmax=True):
        super().__init__()
        self.W1 = nn.Linear(in_features, in_features, bias)
        self.W2 = nn.Linear(in_features, in_features, bias)
        self.uw = nn.Linear(in_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, mask: torch.Tensor):
        u1 = self.W1(h1)
        u2 = self.W2(h2)
        u = u1 + u2
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze()
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class LocationMaskLayer(nn.Module):
    """
    2017-CIKM-Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network
    """

    def __init__(self, location_num, configuration):
        super().__init__()
        self.location_num = location_num
        self.configuration = configuration

    def forward(self, alpha: torch.Tensor):
        location_num = self.location_num
        location_matrix = torch.zeros([location_num, location_num], dtype=torch.float,
                                      device=self.configuration['device'],
                                      requires_grad=False)
        for i in range(location_num):
            for j in range(location_num):
                location_matrix[i, j] = 1 - (abs(i - j) / location_num)
        result = alpha.mm(location_matrix)
        return result


class TextAspectInSentimentOutModel(Model):

    def __init__(self, vocab: Vocabulary, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab)
        self.category_loss_weight = category_loss_weight
        self.sentiment_loss_weight = sentiment_loss_weight

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def no_grad_for_acd_parameter(self):
        pass


class Heat(TextAspectInSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        lstm_input_size = word_embedding_dim
        num_layers = 1
        hidden_size = 32
        self.aspect_gru = torch.nn.GRU(lstm_input_size, hidden_size, batch_first=True,
                                       bidirectional=True, num_layers=num_layers)
        self.sentiment_gru = torch.nn.GRU(lstm_input_size, hidden_size, batch_first=True,
                                          bidirectional=True, num_layers=num_layers)
        self.aspect_attention = AttentionInHtt(hidden_size * 3, hidden_size)
        self.sentiment_attention = AttentionInHtt(hidden_size * 5, hidden_size, softmax=False)

        self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size * 3, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], aspect: torch.Tensor, label: torch.Tensor,
                sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        aspect_gru_output, _ = self.aspect_gru(word_embeddings)
        sentiment_gru_output, _ = self.sentiment_gru(word_embeddings)

        aspect_embeddings_single = self.aspect_embedder(aspect).squeeze(1)
        aspect_repeat = {'aspect': aspect['aspect'].expand_as(tokens['tokens'])}
        aspect_embeddings = self.aspect_embedder(aspect_repeat)

        input_for_aspect_attention = torch.cat([aspect_embeddings, aspect_gru_output], dim=-1)

        aspect_alpha = self.aspect_attention(input_for_aspect_attention, mask)
        category_output = self.element_wise_mul(aspect_gru_output, aspect_alpha, return_not_sum_result=False)

        category_output_unsqueeze = category_output.unsqueeze(1)
        category_output_repeat = category_output_unsqueeze.repeat(1, sentiment_gru_output.size()[1], 1)
        input_for_sentiment_attention = torch.cat([aspect_embeddings, category_output_repeat, sentiment_gru_output],
                                                  dim=-1)
        similarities = self.sentiment_attention(input_for_sentiment_attention, mask)

        location_mask_layer = LocationMaskLayer(aspect_alpha.size(1), self.configuration)
        location_mask = location_mask_layer(aspect_alpha)

        similarities_with_location = similarities * location_mask
        sentiment_alpha = allennlp_util.masked_softmax(similarities_with_location, mask)

        sentiment_output = self.element_wise_mul(sentiment_gru_output, sentiment_alpha, return_not_sum_result=False)

        sentiment_output_with_aspect_embeddings = torch.cat([aspect_embeddings_single, sentiment_output],
                                                            dim=-1)
        final_sentiment_output = self.sentiment_fc(sentiment_output_with_aspect_embeddings)
        final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
        output = {'final_sentiment_output_prob': final_sentiment_output_prob}
        if label is not None:
            sentiment_loss = self.sentiment_loss(final_sentiment_output, label)
            self._accuracy(final_sentiment_output, label)
            output['loss'] = sentiment_loss

        # visualize attention
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [self.categories[sample[i][1][0]]] * 2

                visual_attentions = [aspect_alpha[i][: len(words)].detach().numpy()]
                visual_attentions.extend([sentiment_alpha[i][: len(words)].detach().numpy()])
                titles = ['aspect-true: %s - pred: %s - %s' % (str(label[i].detach().numpy()),
                                                           str(final_sentiment_output_prob[i].detach().numpy()),
                                                           str(self.polarites)
                                                           )
                          ]
                titles.append('sentiment')
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
        }
        return metrics


class AtaeLstm(TextAspectInSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        aspect_word_embedding_dim = aspect_embedder.get_output_dim()
        if self.configuration['model_name'] in ['ae-lstm', 'atae-lstm']:
            lstm_input_size = word_embedding_dim + aspect_word_embedding_dim
        else:
            lstm_input_size = word_embedding_dim
        num_layers = 1
        hidden_size = 300
        self.lstm = torch.nn.LSTM(lstm_input_size, hidden_size, batch_first=True,
                                          bidirectional=False, num_layers=num_layers)
        if self.configuration['model_name'] in ['at-lstm', 'atae-lstm']:
            attention_input_size = word_embedding_dim + aspect_word_embedding_dim
            self.sentiment_attention = AttentionInHtt(attention_input_size, lstm_input_size)
            self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size * 2, self.polarity_num))
        else:
            self.sentiment_attention = None
            self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], aspect: torch.Tensor, label: torch.Tensor,
                sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)

        aspect_embeddings_single = self.aspect_embedder(aspect).squeeze(1)
        aspect_repeat = {'aspect': aspect['aspect'].expand_as(tokens['tokens'])}
        aspect_embeddings = self.aspect_embedder(aspect_repeat)

        if self.configuration['model_name'] in ['ae-lstm', 'atae-lstm']:
            lstm_input = torch.cat([aspect_embeddings, word_embeddings], dim=-1)
        else:
            lstm_input = word_embeddings
        lstm_output, (lstm_hn, lstm_cn) = self.lstm(lstm_input)
        lstm_hn = lstm_hn.squeeze(dim=0)

        if self.configuration['model_name'] in ['at-lstm', 'atae-lstm']:
            input_for_attention = torch.cat([aspect_embeddings, lstm_output], dim=-1)
            alpha = self.sentiment_attention(input_for_attention, mask)
            sentiment_output = self.element_wise_mul(lstm_output, alpha, return_not_sum_result=False)
            sentiment_output = torch.cat([sentiment_output, lstm_hn], dim=-1)
        else:
            sentiment_output = lstm_hn
        final_sentiment_output = self.sentiment_fc(sentiment_output)
        final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
        output = {'final_sentiment_output_prob': final_sentiment_output_prob}
        if label is not None:
            sentiment_loss = self.sentiment_loss(final_sentiment_output, label)
            self._accuracy(final_sentiment_output, label)
            output['loss'] = sentiment_loss

        # visualize attention
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [self.categories[sample[i][1][0]]]

                visual_attentions = [alpha[i][: len(words)].detach().numpy()]
                titles = ['true: %s - pred: %s - %s' % (str(label[i].detach().numpy()),
                                                               str(final_sentiment_output_prob[i].detach().numpy()),
                                                               str(self.polarites)
                                                               )
                          ]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
        }
        return metrics


class Estimator:

    def estimate(self, ds: Iterable[Instance]) -> dict:
        raise NotImplementedError('estimate')


class TextAspectInSentimentOutEstimator:
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self._accuracy = metrics.CategoricalAccuracy()
        self.cuda_device = cuda_device

    def estimate(self, ds: Iterable[Instance]) -> np.ndarray:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            labels = []
            preds = []
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                label = batch['label']
                out_dict = self.model(**batch)
                sentiment_prob = out_dict['final_sentiment_output_prob']
                labels.append(label)
                preds.append(sentiment_prob)
            label_final = torch.cat(labels, dim=0)
            pred_final = torch.cat(preds, dim=0)
            self._accuracy(pred_final, label_final)
        return {'sentiment_acc': self._accuracy.get_metric(reset=True)}


