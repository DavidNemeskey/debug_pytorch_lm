#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Implements the small model from Zaremba (2014). Its main purpose is to
reproduce the numbers exactly, which the implementation in the Pytorch
examples/word_language_model cannot (even if the LR schedule mirrors Zaremba's).
Since it is possible to arrive at the numbers in the paper with a tensorflow
implementation with dynamic_rnn(), it should be possible to do so using
Pytorch as well. Hence this repository; to find out what's wrong with Pytorch.
"""

import argparse

from torch.autograd import Variable


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Modification of the PyTorch Wikitext-2 RNN/LSTM Language '
                    'Model, so that it actually does what Zaremba (2014) '
                    'described.')
    parser.add_argument('--data', '-d', type=str, default='./data/wikitext-2',
                        help='location of the data corpus (files called '
                             'train|valid|test.txt).')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='the model key name.')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    return parser.parse_args()


def batchify(data, bsz, cuda):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    batch processing.
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data


def get_batch(source, i, bptt, evaluation=False):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len])  # .view(-1))
    return data, target


def main():
    args = parse_arguments()


if __name__ == '__main__':
    main()
