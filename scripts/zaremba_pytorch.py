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
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from pytorch_lm import data
from pytorch_lm.lstm_pytorch import Lstm


class SmallZarembaModel(nn.Module):
    """"Implements the small model from Zaremba (2014)."""
    def __init__(self, vocab_size):
        super(SmallZarembaModel, self).__init__()
        self.hidden_size = 200
        self.input_size = 200
        self.num_layers = 2

        self.encoder = nn.Embedding(vocab_size, self.hidden_size)
        self.rnn = Lstm(self.input_size, self.hidden_size, self.num_layers)
        self.decoder = nn.Linear(self.hidden_size, vocab_size)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.uniform_(-initrange, initrange)  # fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        print('INPUT', input.size())
        emb = self.encoder(input)
        print('EMB', emb.size())
        # self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Modification of the PyTorch Wikitext-2 RNN/LSTM Language '
                    'Model, so that it actually does what Zaremba (2014) '
                    'described.')
    parser.add_argument('--data', '-d', type=str, default='./data/wikitext-2',
                        help='location of the data corpus (files called '
                             'train|valid|test.txt).')
    parser.add_argument('--model', '-m', type=str, default='LSTM',
                        help='the model key name.')
    parser.add_argument('--seed', '-s', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', '-c', action='store_true', help='use CUDA')
    parser.add_argument('--log-interval', '-l', type=int, default=200, metavar='N',
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
    data = data.narrow(0, 0, nbatch * bsz).contiguous()
    # Evenly divide the data across the bsz batches.
    # data = data.view(bsz, -1).t().contiguous()
    data = data.view(bsz, -1)
    if cuda:
        data = data.cuda()
    print('DS', data.size())
    return data


def get_batch(source, i, num_steps, evaluation=False):
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
    seq_len = min(num_steps, source.size(1) - 1 - i)
    # TODO can we no_grad target as well?
    data_chunk = source[:, i:i+seq_len].contiguous()
    target_chunk = source[:, i+1:i+1+seq_len].contiguous()  # .view(-1))
    if evaluation:
        with torch.no_grad():
            data = Variable(data_chunk)
    else:
        data = Variable(data_chunk)
    target = Variable(target_chunk)  # .view(-1))
    return data, target


def train(model, corpus, train_data, criterion, epoch, lr, batch_size,
          num_steps, log_interval):
    # Turn on training mode which enables dropout.
    print('TRAIN', epoch)
    model.train()
    total_loss = 0
    start_time = time.time()
    vocab_size = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, num_steps)):
        print('FOR', batch, i)
        data, targets = get_batch(train_data, i, num_steps)

        def to_str(f):
            return corpus.dictionary.idx2word[f]

        # print(data.data.cpu().numpy())
        # import numpy as np
        # print('DATA\n', np.vectorize(to_str)(data.data.cpu().numpy()))
        # print('TARGET\n', np.vectorize(to_str)(targets.data.cpu().numpy()))
        # print(targets.data.cpu().numpy())
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        # print('TARGETS\n', np.vectorize(to_str)(targets.data.cpu().numpy()))
        # _, indices = output.max(2)
        # print('OUTPUT\n', np.vectorize(to_str)(indices.data.cpu().numpy()))
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm(model.parameters(), config.clip)
        # all_min, all_max, all_sum, all_size = 1000, -1000, 0, 0
        for name, p in model.named_parameters():
            data = p.grad.data
            # shape = data.size()
            # all_min = all_min if data.min() >= all_min else data.min()
            # all_max = all_max if data.max() <= all_max else data.max()
            # all_sum += data.sum()
            # from functools import reduce
            # all_size += reduce(lambda a, b: a * b, shape)
            # print(name, shape, data.min(), data.max(), data.mean(), data.std())
            p.grad.data.clamp_(-5.0, 5.0)
            p.data.add_(-1 * lr, p.grad.data)
        # print('Sum', all_min, all_max, all_sum / all_size)
        # print()
        # if batch % log_interval == 0 and batch > 0:
        #     sys.exit()

        total_loss += loss.data

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                  'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // num_steps, lr,
                      elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)),
                  flush=True)
            total_loss = 0
            start_time = time.time()


def evaluate(model, corpus, data_source, criterion, batch_size, num_steps):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    vocab_size = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, num_steps):
        data, targets = get_batch(data_source, i, num_steps, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        total_loss += len(data) * criterion(output_flat, targets_flat).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return [repackage_hidden(v) for v in h]


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably '
                  'run with --cuda')
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data)

    train_batch_size = 20
    eval_batch_size = 20
    num_steps = 20
    train_data = batchify(corpus.train, train_batch_size, args.cuda)
    val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
    test_data = batchify(corpus.test, eval_batch_size, args.cuda)

    ###############################################################################
    # Build the model
    ###############################################################################

    vocab_size = len(corpus.dictionary)
    model = SmallZarembaModel(vocab_size)
    if args.cuda:
        model.cuda()

    ###############################################################################
    # Training code
    ###############################################################################

    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    orig_lr = 1.0
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, 13 + 1):
            lr_decay = 0.5 ** max(epoch - 4, 0.0)
            lr = orig_lr * lr_decay
            epoch_start_time = time.time()
            train(model, corpus, train_data, criterion, epoch,
                  lr, train_batch_size, num_steps, args.log_interval)
            val_loss = evaluate(model, corpus, val_data,
                                criterion, eval_batch_size, num_steps)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(model, corpus, test_data,
                         criterion, eval_batch_size, num_steps)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


if __name__ == '__main__':
    main()
