# ACSA
> Papers, models and datasets for Aspect-Category Sentiment Analysis.

![LICENSE](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

## Requirement
* python 3.6
* pytorch 1.3.0
* allennlp 0.9.0

## Usage

### Supported datasets
- [SemEval-2014 Task 4](http://alt.qcri.org/semeval2014/task4/)
    - SemEval-2014-Task-4-LAPT
    - SemEval-2014-Task-4-REST
- [2018-aaai-Learning to Attend via Word-Aspect Associative Fusion for Aspect-based Sentiment Analysis](https://arxiv.org/abs/1712.05403v1)
    - SemEval-2014-Task-4-REST-DevSplits
- [2018-acl-Aspect Based Sentiment Analysis with Gated Convolutional Networks](https://arxiv.org/abs/1805.07043)
    - SemEval-2014-Task-4-REST-Hard
    - SemEval-141516-LARGE-REST-HARD
- [SemEval-2015 Task 12](http://alt.qcri.org/semeval2015/task12/)
    - SemEval-2015-Task-12-LAPT
    - SemEval-2015-Task-12-REST
    - SemEval-2015-Task-12-HOTEL
- [SemEval-2016 Task 5](http://alt.qcri.org/semeval2016/task5/)
    - SemEval-2016-Task-5-CH-CAME-SB1
    - SemEval-2016-Task-5-CH-PHNS-SB1
    - SemEval-2016-Task-5-LAPT-SB1
    - SemEval-2016-Task-5-LAPT-SB2
    - SemEval-2016-Task-5-REST-SB1
    - SemEval-2016-Task-5-REST-SB2
- [2019-emnlp-A Challenge Dataset and Effective Models for Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/D19-1654.pdf)
    - MAMSACSA
    - MAMSATSA

### Aspect-Category Sentiment Classification (ACSC) Models
#### Supported models
- ae-lstm [2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification](https://www.aclweb.org/anthology/D16-1058.pdf)
![ae-lstm](images/ae-lstm.png)
- at-lstm [2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification](https://www.aclweb.org/anthology/D16-1058.pdf)
![at-lstm](images/at-lstm.png)
- atae-lstm [2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification](https://www.aclweb.org/anthology/D16-1058.pdf)
![atae-lstm](images/atae-lstm.png)
- Heat(papers/2017-CIKM-Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network.pdf)
![heat](images/heat.png)
#### Training
sh scripts/run.sh acsa/acac_pytorch/acsc_bootstrap.py --model_name Heat --current_dataset SemEval-2014-Task-4-REST-DevSplits --embedding_filepath glove.840B.300d.txt

#### Visualization
sh scripts/run.sh acsa/acac_pytorch/acsc_bootstrap.py --model_name Heat --current_dataset SemEval-2014-Task-4-REST-DevSplits --embedding_filepath glove.840B.300d.txt --train False --visualize_attention True

### Aspect Category Detection (ACD) Models

### Joint Models for ACSC and ACD

## Paper
Suggestions about adding papers are welcomed!
- 2016-emnlp-A Hierarchical Model of Reviews for Aspect-based Sentiment Analysis
- 2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification
- 2017-CIKM-Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network
- 2017-wsdm-Deep Memory Networks for Attitude Identification
- 2018-aaai-Learning to Attend via Word-Aspect Associative Fusion for Aspect-based Sentiment Analysis
- 2018-acl-Aspect Based Sentiment Analysis with Gated Convolutional Networks
- 2018-emnlp-Joint Aspect and Polarity Classification for Adpect-based Sentiment Analysis with End-to-End Neural Networks
- 2019-AAAI-A Human-Like Semantic Cognition Network for Aspect-Level Sentiment Classification
- 2019-conll-Learning to Detect Opinion Snippet for Aspect-Based Sentiment Analysis
- 2019-emnlp-A Challenge Dataset and Effective Models for Aspect Based Sentiment Analysis
- 2019-emnlp-A Novel Aspect-Guided Deep Transition Model for Aspect Based Sentiment Analysis
- 2019-emnlp-CAN Constrained Attention Networks for Multi-Aspect Sentiment Analysis
- 2019-ijcai-Earlier Attention Aspect-Aware LSTM for Aspect Sentiment Analysis
- 2019-naacl-Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence
- 2019-tkdd-Aspect Aware Learning for Aspect Category Sentiment Analysis
- 2019-www-Aspect-level Sentiment Analysis using AS-Capsules

## Contributions

Feel free to contribute!

You can raise an issue or submit a pull request, whichever is more convenient for you.

## Licence

MIT License