## Dataset Splits for our AAAI'18 paper

Note:The datasets have been updated because the wrong ones have been uploaded. Only Laptop Term and Restaurant Aspects should be affected. The rest should be fine. Drop me an email if you find any issues.

Commonly used SemEval14 datasets (Laptop and Restaurants) do not use
dev splits. In many works, the evaluation is conducted *solely* on the test set. We believe that proper evaluation should include a dev set.

In this repository, you will find the train/test/dev splits from our
AAAI'18 paper **Learning to attend via Word-Aspect Associative Fusion for
Aspect-based Sentiment Analysis**.

SE14+15 dataset was obtained from Li et al. 2017's "Deep Memory Networks
for Attitude Identification, WSDM'17"

## Usage

You will find 4 `.pkl` files in the `/dataset` directory :smiley:. It is a `dict`object with several keys such as word_index (and index_word) indexes.
Train, dev, and val are found as "training",'dev' and "test" respectively.

The format of each data instance is:

```
[tokenized_txt, actual_len, tokenized_terms, term_len, polarity, info]
```
where `info` is something like the position information.
