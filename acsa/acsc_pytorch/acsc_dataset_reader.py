import copy

from typing import *
from overrides import overrides
import pickle
import copy

from allennlp.data.fields import TextField, MetadataField, ArrayField, ListField, LabelField, MultiLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import DatasetReader
import spacy
from nltk.corpus import stopwords
english_stop_words = stopwords.words('english')
english_stop_words.extend([',', '.', '?', ';', '-', ':', '\'', '"', '(', ')', '!'])

from acsa.utils import corenlp_factory
from acsa.utils import create_graph
from acsa.utils import my_corenlp


class TextAspectInSentimentOut(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 aspect_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.aspect_indexers = aspect_indexers or {"aspect": SingleIdTokenIndexer(namespace='aspect')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample[0].strip()

        words = self.tokenizer(text)
        sample.append(words)
        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        aspect = [Token(self.categories[sample[1][0]])]
        aspect_field = TextField(aspect, self.aspect_indexers)
        fields['aspect'] = aspect_field

        fields["label"] = LabelField(sample[1][1], skip_indexing=True)
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            for aspect in sample[1]:
                yield self.text_to_instance([sample[0], aspect])