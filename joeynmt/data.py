# coding: utf-8
"""
Data module
"""
import io
import sys
import random
import os
import os.path
from typing import Optional
import logging

# pylint: disable=no-name-in-module
from torchtext.legacy.datasets import TranslationDataset
from torchtext.legacy.data import Dataset, Iterator, Field, Example

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary

logger = logging.getLogger(__name__)


def load_data(data_cfg: dict, datasets: list = None)\
        -> (Dataset, Dataset, Optional[Dataset], Vocabulary, Vocabulary, Optional[Vocabulary]):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :param datasets: list of dataset names to load
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
        - factor_vocab: factor vocabulary extracted from source training data
    """
    if datasets is None:
        datasets = ["train", "dev", "test"]

    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg.get("train", None)
    dev_path = data_cfg.get("dev", None)
    test_path = data_cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError('Please specify at least one data source path.')

    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]
    use_factor = data_cfg.get("use_factor", False)

    if level == "char":
        assert not use_factor, "Character-level translation incompatible with source factors."

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = Field(init_token=None, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
                      tokenize=tok_fun, batch_first=True, lower=lowercase,
                      unk_token=UNK_TOKEN, include_lengths=True)

    trg_field = Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                      pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
                      tokenize=tok_fun, batch_first=True, lower=lowercase,
                      include_lengths=True)

    factor_field = Field(init_token=None, eos_token=EOS_TOKEN,
                         pad_token=PAD_TOKEN, tokenize=tok_fun,
                         unk_token=UNK_TOKEN,
                         batch_first=True, lower=lowercase,
                         include_lengths=True)

    train_data = None
    if "train" in datasets and train_path is not None:
        logger.info("Loading training data...")

        if use_factor:
            exts = ("." + src_lang, "." + trg_lang, ".factor",)
            fields = (src_field, trg_field, factor_field,)

            dataset_class = FactoredTranslationDataset
        else:
            exts = ("." + src_lang, "." + trg_lang,)
            fields = (src_field, trg_field,)

            dataset_class = TranslationDataset

        train_data = dataset_class(path=train_path,
                                   exts=exts,
                                   fields=fields,
                                   filter_pred=
                                   lambda x: len(vars(x)['src'])
                                   <= max_sent_length
                                   and len(vars(x)['trg'])
                                   <= max_sent_length)

        random_train_subset = data_cfg.get("random_train_subset", -1)
        if random_train_subset > -1:
            # select this many training examples randomly and discard the rest
            keep_ratio = random_train_subset / len(train_data)
            keep, _ = train_data.split(
                split_ratio=[keep_ratio, 1 - keep_ratio],
                random_state=random.getstate())
            train_data = keep

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    factor_max_size = data_cfg.get("factor_voc_limit", sys.maxsize)
    factor_min_freq = data_cfg.get("factor_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)
    factor_vocab_file = data_cfg.get("factor_vocab", None)

    assert (train_data is not None) or (src_vocab_file is not None)
    assert (train_data is not None) or (trg_vocab_file is not None)

    logger.info("Building vocabulary...")
    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)
    if use_factor:
        factor_vocab = build_vocab(field="factor", min_freq=factor_min_freq,
                                max_size=factor_max_size,
                                dataset=train_data, vocab_file=factor_vocab_file)
    else:
        factor_vocab = None

    dev_data = None
    if "dev" in datasets and dev_path is not None:
        logger.info("Loading dev data...")
        dev_data = TranslationDataset(path=dev_path,
                                      exts=("." + src_lang, "." + trg_lang),
                                      fields=(src_field, trg_field))

    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("Loading test data...")
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = dataset_class(
                path=test_path, exts=exts,
                fields=fields)
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    factor_field.vocab = factor_vocab

    logger.info("Data loaded.")
    return train_data, dev_data, test_data, src_vocab, trg_vocab, factor_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = Iterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = Iterator(
            repeat=False, dataset=dataset, batch_size=batch_size,
            batch_size_fn=batch_size_fn, train=False, sort=False)

    return data_iter


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            # Because the file can be stdin which doesn't need to be opened,
            # easier not to use with here.
            # pylint: disable=bad-option-value,consider-using-with
            src_file = open(src_path, encoding="utf-8")

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(Example.fromlist([src_line], fields))

        src_file.close()

        super().__init__(examples, fields, **kwargs)


class MonoFactorDataset(Dataset):
    """Defines a dataset for factored machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, src_field: Field, factor_field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.
        Assumes that each line of the input file is structured as follows:

        source words ||| factors for all source words

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param fields: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', src_field),
                  ('factor', factor_field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()

            # assuming ||| separator between source and factors
            parts = [part.strip() for part in src_line.split("|||")]

            src_part = parts[0]
            factor_part = parts[1]

            if src_part != '' and factor_part != '':
                examples.append(data.Example.fromlist(
                    [src_part, factor_part], fields))

        src_file.close()

        super(MonoFactorDataset, self).__init__(examples, fields, **kwargs)


class FactoredTranslationDataset(Dataset):
    """Defines a dataset for factored machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages, plus source factors.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('factor', fields[2])]

        src_path, trg_path, factor_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
                io.open(factor_path, mode='r', encoding='utf-8') as factor_file:
                for src_line, trg_line, factor_line in zip(src_file, trg_file, factor_file):
                    src_line, trg_line, factor_line = src_line.strip(), trg_line.strip(), factor_line.strip()
                    if src_line != '' and trg_line != '' and factor_line != '':
                        examples.append(data.Example.fromlist(
                            [src_line, trg_line, factor_line], fields))

        super(FactoredTranslationDataset, self).__init__(examples, fields, **kwargs)

    # pylint: disable=arguments-differ
    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.
        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
