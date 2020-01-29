# -*- coding: utf-8 -*-

from allennlp.models import Model

from acsa.acsc_pytorch import acsc_models


class Callback(object):
    """Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def on_epoch_end(self, epoch: int):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_batch_end(self, batch: int):
        pass

    def on_train_begin(self):
        pass


class EstimateCallback(Callback):

    def __init__(self, data_type_and_data: dict, estimator: acsc_models.Estimator, logger):
        self.data_type_and_data = data_type_and_data
        self.estimator = estimator
        self.logger = logger

    def on_epoch_end(self, epoch):
        for data_type, data in self.data_type_and_data.items():
            result = self.estimator.estimate(data)
            self.logger.info('epoch: %d data_type: %s result: %s' % (epoch, data_type, str(result)))

    def on_batch_end(self, batch: int):
        for data_type, data in self.data_type_and_data.items():
            result = self.estimator.estimate(data)
            self.logger.info('batch: %d data_type: %s result: %s' % (batch, data_type, str(result)))
