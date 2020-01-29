import sys
import os

import torch
from allennlp.data.iterators import BucketIterator
from allennlp.data.iterators import BasicIterator
from allennlp.modules.text_field_embedders import TextFieldEmbedder
import torch.optim as optim
from acsa.acsc_pytorch.my_allennlp_trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer

from acsa.acsc_pytorch import acsc_models
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from acsa.acsc_pytorch import allennlp_callback
from acsa.model_train_template.model_train_template import ModelTrainTemplate
from acsa.acsc_pytorch import acsc_dataset_reader


class TextAspectInSentimentOutTrainTemplate(ModelTrainTemplate):

    def __init__(self, configuration):
        super().__init__(configuration)
        self.data_reader = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.distinct_categories = None
        self.distinct_polarities = None
        self._load_data()

        self.vocab = None
        self._build_vocab()

        self.iterator = None
        self.val_iterator = None
        self._build_iterator()

    def _load_data(self):
        data_filepath = self.base_data_dir + 'data'
        if os.path.exists(data_filepath):
            self.train_data, self.dev_data, self.test_data, self.distinct_categories, self.distinct_polarities, \
                = super()._load_object(data_filepath)
        else:
            train_dev_test_data, distinct_categories, distinct_polarities = self.dataset. \
                generate_acd_and_sc_data()

            distinct_polarities_new = []
            for polarity in distinct_polarities:
                if polarity != 'conflict':
                    distinct_polarities_new.append(polarity)
            self.distinct_categories = distinct_categories
            self.distinct_polarities = distinct_polarities_new

            token_indexer = SingleIdTokenIndexer(namespace="tokens",
                                                 token_min_padding_length=self.configuration['token_min_padding_length'])
            aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
            reader = acsc_dataset_reader.TextAspectInSentimentOut(
                self.distinct_categories, self.distinct_polarities,
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                aspect_indexers={'aspect': aspect_indexer},
                configuration=self.configuration
            )
            self.data_reader = reader

            train_dev_test_data_label_indexed = {}
            for data_type, data in train_dev_test_data.items():
                if data is None:
                    continue
                data_new = []
                for sample in data:
                    sample_new = [sample[0]]
                    labels_new = []
                    for label in sample[1]:
                        aspect = label[0]
                        polarity = label[1]
                        aspect_index = distinct_categories.index(aspect)
                        if polarity == 'conflict':
                            polarity_index = -100
                        else:
                            polarity_index = distinct_polarities_new.index(polarity)
                        labels_new.append((aspect_index, polarity_index))
                    if len(labels_new) != 0:
                        sample_new.append(labels_new)
                        data_new.append(sample_new)
                train_dev_test_data_label_indexed[data_type] = data_new
            self.train_data = reader.read(train_dev_test_data_label_indexed['train'])
            self.dev_data = reader.read(train_dev_test_data_label_indexed['dev'])
            self.test_data = reader.read(train_dev_test_data_label_indexed['test'])
            data = [self.train_data, self.dev_data, self.test_data, self.distinct_categories,
                    self.distinct_polarities]
            super()._save_object(data_filepath, data)

    def _build_vocab(self):
        if self.configuration['train']:
            vocab_file_path = self.base_data_dir + 'vocab'
            if os.path.exists(vocab_file_path):
                self.vocab = super()._load_object(vocab_file_path)
            else:
                data = self.train_data + self.dev_data + self.test_data
                self.vocab = Vocabulary.from_instances(data, max_vocab_size=sys.maxsize)
                super()._save_object(vocab_file_path, self.vocab)
            self.model_meta_data['vocab'] = self.vocab
        else:
            self.vocab = self.model_meta_data['vocab']

    def _build_iterator(self):
        self.iterator = BucketIterator(batch_size=self.configuration['batch_size'],
                                       sorting_keys=[("tokens", "num_tokens")],
                                       )
        self.iterator.index_with(self.vocab)
        self.val_iterator = BasicIterator(batch_size=self.configuration['batch_size'])
        self.val_iterator.index_with(self.vocab)

    def _print_args(self, model):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('> training arguments:')
        for arg in self.configuration.keys():
            self.logger.info('>>> {0}: {1}'.format(arg, self.configuration[arg]))

    def _find_model_function_pure(self):
        raise NotImplementedError('_find_model_function_pure')

    def _get_aspect_embeddings_dim(self):
        return 300

    def _init_aspect_embeddings_from_word_embeddings(self):
        return False

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        aspect_embedding_matrix = None
        if self._init_aspect_embeddings_from_word_embeddings():
            embedding_filepath = self.configuration['embedding_filepath']
            aspect_embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                                self.vocab, namespace='aspect')
        aspect_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='aspect'),
                                    embedding_dim=self._get_aspect_embeddings_dim(), padding_index=0,
                                     trainable=True, weight=aspect_embedding_matrix)
        aspect_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"aspect": aspect_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)
        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            aspect_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration
        )
        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_estimator(self, model):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = acsc_models.TextAspectInSentimentOutEstimator(model, self.val_iterator,
                                                                     self.distinct_categories,
                                                                     self.distinct_polarities,
                                                                     cuda_device=gpu_id)
        return estimator

    def _get_estimate_callback(self, model):
        result = []
        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }

        estimator = self._get_estimator(model)
        estimate_callback = allennlp_callback.EstimateCallback(data_type_and_data, estimator, self.logger)
        result.append(estimate_callback)
        return result

    def _inner_train(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        self.model = self._find_model_function()
        # optimizer = adagrad.Adagrad(self.model.parameters(), lr=0.01, weight_decay=0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.00001)

        callbacks = self._get_estimate_callback(self.model)
        early_stopping_by_batch: bool = False
        estimator = self._get_estimator(self.model)

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            iterator=self.iterator,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data,
            cuda_device=gpu_id,
            num_epochs=self.configuration['epochs'],
            validation_metric='+accuracy',
            validation_iterator=self.val_iterator,
            serialization_dir=self.model_dir,
            patience=self.configuration['patience'],
            callbacks =callbacks,
            early_stopping_by_batch=early_stopping_by_batch,
            estimator=estimator
        )
        metrics = trainer.train()
        self.logger.info('metrics: %s' % str(metrics))

    def _save_model(self):
        torch.save(self.model, self.best_model_filepath)

    def _load_model(self):
        self.model = torch.load(self.best_model_filepath)
        self.model.configuration = self.configuration

    def evaluate(self):
        estimator = self._get_estimator(self.model)

        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        for data_type, data in data_type_and_data.items():
            result = estimator.estimate(data)
            self.logger.info('data_type: %s result: %s' % (data_type, result))


class Heat(TextAspectInSentimentOutTrainTemplate):
    """
    2017-CIKM-Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_aspect_embeddings_dim(self):
        return 32

    def _find_model_function_pure(self):
        return acsc_models.Heat


class AtaeLstm(TextAspectInSentimentOutTrainTemplate):
    """
    2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _find_model_function_pure(self):
        return acsc_models.AtaeLstm
