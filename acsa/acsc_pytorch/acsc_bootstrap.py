import argparse
import sys
import random

import torch
import numpy

from acsa.utils import argument_utils
from acsa.acsc_pytorch import acsc_templates

parser = argparse.ArgumentParser()
parser.add_argument('--current_dataset', help='dataset name', default='SemEval-2014-Task-4-REST-DevSplits', type=str)
parser.add_argument('--task_name', help='task name', default='acsc', type=str)
parser.add_argument('--model_name', help='model name', default='Heat', type=str)
parser.add_argument('--timestamp', help='timestamp', default=int(1571400646), type=int)
parser.add_argument('--train', help='if train a new model', default=True, type=argument_utils.my_bool)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--epochs', help='epochs', default=100, type=int)
parser.add_argument('--batch_size', help='batch_size', default=64, type=int)
parser.add_argument('--patience', help='patience', default=10, type=int)
parser.add_argument('--visualize_attention', help='visualize attention', default=False, type=argument_utils.my_bool)
parser.add_argument('--embedding_filepath', help='embedding filepath', type=str)
parser.add_argument('--embed_size', help='embedding dimension', default=300, type=int)
parser.add_argument('--seed', default=776, type=int)
parser.add_argument('--device', default=None, type=str)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--token_min_padding_length', default=5, type=int)
parser.add_argument('--debug', default=False, type=argument_utils.my_bool)
parser.add_argument('--early_stopping_by_batch', default=False, type=argument_utils.my_bool)

args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device)
gpu_ids = args.gpu_id.split(',')
if len(gpu_ids) == 1:
    args.gpu_id = -1 if int(gpu_ids[0]) == -1 else 0
else:
    args.gpu_id = list(range(len(gpu_ids)))

configuration = args.__dict__

data_name = args.current_dataset

if configuration['seed'] is not None:
    random.seed(configuration['seed'])
    numpy.random.seed(configuration['seed'])
    torch.manual_seed(configuration['seed'])
    torch.cuda.manual_seed(configuration['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

for i in range(args.repeat):
    configuration['model_name_complete'] = '%s-%d' % (args.model_name, args.repeat)

    model_name = configuration['model_name']
    if model_name in ['ae-lstm', 'at-lstm', 'atae-lstm']:
        template = acsc_templates.AtaeLstm(configuration)
    elif model_name == 'Heat':
        template = acsc_templates.Heat(configuration)
    else:
        raise NotImplementedError(model_name)

    if configuration['train']:
        template.train()
    template.evaluate()
