# coding: utf-8
"""
Embedding module
"""

import io
import math
import logging
import torch
from torch import nn, Tensor
from joeynmt.helpers import freeze_params
from joeynmt.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = False,
                 **kwargs) -> None:
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim,
                                padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)

    #from fairseq
    def load_from_file(self, embed_path: str, vocab: Vocabulary):
        """Load pretrained embedding weights from text file.

        - First line is expected to contain vocabulary size and dimension.
          The dimension has to match the model's specified embedding size,
          the vocabulary size is used in logging only.
        - Each line should contain word and embedding weights
          separated by spaces.
        - The pretrained vocabulary items that are not part of the
          joeynmt's vocabulary will be ignored (not loaded from the file).
        - The initialization (specified in config["model"]["embed_initializer"])
          of joeynmt's vocabulary items that are not part of the
          pretrained vocabulary will be kept (not overwritten in this func).
        - This function should be called after initialization!

        Example:
            2 5
            the -0.0230 -0.0264  0.0287  0.0171  0.1403
            at -0.0395 -0.1286  0.0275  0.0254 -0.0932

        :param embed_path: embedding weights text file
        :param vocab: Vocabulary object
        """
        embed_dict = {}
        # parse file
        with io.open(embed_path, 'r', encoding='utf-8',
                     errors='ignore') as f_embed:
            vocab_size, d = map(int, f_embed.readline().split())
            assert self.embedding_dim == d, \
                "Embedding dimension doesn't match."
            for line in f_embed.readlines():
                tokens = line.rstrip().split(' ')
                if tokens[0] in vocab.stoi.keys():
                    embed_dict[tokens[0]] = torch.FloatTensor(
                        [float(t) for t in tokens[1:]])

            logger.warning("Loaded {} of {} ({:%}) tokens "
                           "in the pre-trained embeddings.".format(
                           len(embed_dict), vocab_size,
                           len(embed_dict)/vocab_size))

        # assign
        for idx in range(len(vocab)):
            token = vocab.itos[idx]
            if token in embed_dict:
                assert self.embedding_dim == len(embed_dict[token])
                self.lut.weight.data[idx] = embed_dict[token]

        logger.warning("Loaded {} of {} ({:%}) tokens "
                       "of the JoeyNMT's vocabulary.".format(
                        len(embed_dict), len(vocab),
                        len(embed_dict)/len(vocab)))

def concatenate_embeddings(src_embedded: Tensor, factor_embedded: Tensor) -> Tensor:
    """
    Concatenate embeddings to combine source words and their factors.

    :param src_embedded: embedded src inputs,
        shape (batch_size, src_len, src_embed_size)
    :param factor_embedded:
        shape (batch_size, src_len, factor_embed_size)
    :return:
    """

    # check shapes
    assert src_embedded.shape[0] == factor_embedded.shape[0]
    assert src_embedded.shape[1] == factor_embedded.shape[1]

    concat_embedded = torch.cat((src_embedded, factor_embedded), -1)

    return concat_embedded


def sum_embeddings(src_embedded: Tensor, factor_embedded: Tensor) -> Tensor:
    """
    Sum embeddings to combine source words and their factors.

    :param src_embedded: embedded src inputs,
        shape (batch_size, src_len, src_embed_size)
    :param factor_embedded:
        shape (batch_size, src_len, factor_embed_size)
    :return:
    """

    # check shapes
    assert src_embedded.shape[0] == factor_embedded.shape[0]
    assert src_embedded.shape[1] == factor_embedded.shape[1]
    assert src_embedded.shape[2] == factor_embedded.shape[2]

    sum_embedded = src_embedded + factor_embedded

    return sum_embedded
