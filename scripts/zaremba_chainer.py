#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Implements the small model from Zaremba (2014). Its main purpose is to
reproduce the numbers exactly.
"""

import argparse
from functools import partial
import math
import os
import sys
import time

import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.links as L
import chainer.functions as F

from pytorch_lm.data import Corpus
from pytorch_lm.lstm_chainer import Lstm
from pytorch_lm.loss import chainer_sequence_loss


class SmallZarembaModel(Chain):
    """"Implements the small model from Zaremba (2014)."""
    def __init__(self, vocab_size):
        self.hidden_size = 200
        self.input_size = 200
        self.num_layers = 2

        encoder = L.EmbedID(vocab_size, self.input_size)
        rnn = Lstm(self.input_size, self.hidden_size, self.num_layers)
        decoder = L.Linear(self.hidden_size, vocab_size)
        super(SmallZarembaModel, self).__init__(
            encoder=encoder, rnn=rnn, decoder=decoder)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        uniform = chainer.initializers.Uniform(initrange)
        uniform(self.encoder.W.data)
        uniform(self.decoder.W.data)
        uniform(self.decoder.b.data)

    def __call__(self, input, hidden, trace=False):
        # print('INPUT', input.size())
        emb = self.encoder(input)
        # print('EMB', emb.size())
        # self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(F.reshape(output, (-1, output.shape[2])))
        if trace:
            print('EMB', F.copy(emb, -1).data)
            print('RNN_OUT', F.copy(output, -1).data)
        return (
            F.reshape(decoded, (output.shape[0], output.shape[1], decoded.shape[1])),
            hidden
        )

    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)

    def save_parameters(self, out_dict=None, prefix=''):
        if out_dict is None:
            out_dict = {}
        self.rnn.save_parameters(out_dict, prefix=prefix + 'RNN/')
        out_dict[prefix + 'embedding'] = F.copy(self.encoder.W, -1).data
        # .T is required because stupid Linear stores the weights transposed
        out_dict[prefix + 'softmax_w'] = F.copy(self.decoder.W, -1).data.T
        out_dict[prefix + 'softmax_b'] = F.copy(self.decoder.b, -1).data
        return out_dict

    def load_parameters(self, data_dict, prefix=''):
        device_id = self._device_id if self._device_id is not None else -1

        self.rnn.load_parameters(data_dict, prefix=prefix + 'RNN/')
        self.encoder.W.data = F.copy(data_dict[prefix + 'embedding'], device_id).data
        # .T is required because stupid Linear stores the weights transposed
        self.decoder.W.data = F.copy(data_dict[prefix + 'softmax_w'], device_id).data.T
        self.decoder.b.data = F.copy(data_dict[prefix + 'softmax_b'], device_id).data


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
    parser.add_argument('--trace-data', '-T', type=int, default=0,
                        help='log the weights and results of each component. '
                             'Exits after the specified number of minibatches.')
    save_load = parser.add_mutually_exclusive_group()
    save_load.add_argument('--save-params', '-S',
                           help='save parameters to an .npz file and exit.')
    save_load.add_argument('--load-params', '-L',
                           help='load parameters from an .npz file.')
    parser.add_argument('--lr', type=float, default=1.0)
    return parser.parse_args()


def batchify(data, bsz, xp):
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
    nbatch = len(data) // bsz
    rbatch = 20 * ((nbatch - 1) // 20) + 1
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    # data = data.narrow(0, 0, nbatch * bsz).contiguous()
    data = data[:rbatch * bsz]
    # Evenly divide the data across the bsz batches.
    # data = data.view(bsz, -1).t().contiguous()
    # data = data.view(bsz, -1)
    data = data.reshape(bsz, -1)
    return xp.array(data, copy=False)


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
    seq_len = min(num_steps, source.shape[1] - 1 - i)
    # TODO can we no_grad target as well?
    data_chunk = source[:, i:i+seq_len]
    target_chunk = source[:, i+1:i+1+seq_len]
    if evaluation:
        with chainer.no_backprop_mode():
            data = Variable(data_chunk)
    else:
        data = Variable(data_chunk)
    target = Variable(target_chunk)  # .view(-1))
    return data, target


def train(model, corpus, train_data, criterion, epoch, lr, batch_size,
          num_steps, log_interval, trace=0):
    # Turn on training mode which enables dropout.
    chainer.config.train = True
    total_loss = 0
    start_time = time.time()
    data_len = train_data.shape[1]
    hidden = model.init_hidden(batch_size)
    if trace:
        print('HIDDEN', [[F.copy(v, -1).data for v in t] for t in hidden])

    for batch, i in enumerate(range(0, data_len - 1, num_steps)):
        # print('FOR', batch, i, (train_data.size(1) - 1) // num_steps)
        data, targets = get_batch(train_data, i, num_steps)
        if trace:
            print('BATCH', batch)
            print('DATA', F.copy(data, -1).data)
            print('TARGET', F.copy(targets, -1).data)

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
        model.cleargrads()
        output, hidden = model(data, hidden, trace)
        if trace:
            print('LOGITS', output.data.shape, F.copy(output, -1).data)
            print('FINAL STATE', [[F.copy(v, -1).data for v in t] for t in hidden])
        # print('TARGETS\n', np.vectorize(to_str)(targets.data.cpu().numpy()))
        # _, indices = output.max(2)
        # print('OUTPUT\n', np.vectorize(to_str)(indices.data.cpu().numpy()))
        loss = criterion(output, targets)
        # print('TRAIN COST', loss)
        if trace:
            print('LOSS', loss)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm(model.parameters(), config.clip)
        # all_min, all_max, all_sum, all_size = 1000, -1000, 0, 0
        for name, p in model.namedparams():
            # data = p.grad.data
            # shape = data.size()
            # all_min = all_min if data.min() >= all_min else data.min()
            # all_max = all_max if data.max() <= all_max else data.max()
            # all_sum += data.sum()
            # from functools import reduce
            # all_size += reduce(lambda a, b: a * b, shape)
            # print(name, shape, data.min(), data.max(), data.mean(), data.std())
            if trace:
                print('GRAD', name[1:], F.copy(p.grad, -1).data)
            p.grad_var = F.clip(p.grad_var, -5.0, 5.0)
            if trace:
                print('GRAD CLIP', name[1:], F.copy(p.grad, -1).data)
            p.data.add_(-1 * lr, p.grad.data)
            if trace:
                print('NEW VALUE', name[1:], F.copy(p, -1).data)
        # print('Sum', all_min, all_max, all_sum / all_size)
        # print()
        # if batch % log_interval == 0 and batch > 0:
        #     sys.exit()

        if trace and batch == trace - 1:  # batch counts from 0
            print('Trace done; exiting...')
            sys.exit()

        total_loss += loss.data / num_steps

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                  'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, data_len // num_steps, lr,
                      elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)),
                  flush=True)
            total_loss = 0
            start_time = time.time()
            # sys.exit()  # Was here for the LR sweep


def evaluate(model, corpus, data_source, criterion, batch_size, num_steps):
    # Turn on evaluation mode which disables dropout.
    chainer.config.train = False
    total_loss = 0
    data_len = data_source.shape[1]
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_len - 1, num_steps):
        data, targets = get_batch(data_source, i, num_steps, evaluation=True)
        output, hidden = model(data, hidden)
        cost = criterion(output, targets).data
        total_loss += cost
        hidden = repackage_hidden(hidden)
    return total_loss[0] / data_len


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return [repackage_hidden(v) for v in h]


def main():
    args = parse_arguments()
    # I am not convinced this works (maybe should be done before importing
    # chainer, also https://github.com/chainer/chainer/issues/3751)
    os.environ['CHAINER_SEED'] = str(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = Corpus(args.data)

    vocab_size = len(corpus.dictionary)
    model = SmallZarembaModel(vocab_size)
    if args.cuda:
        model.to_gpu()

    print('MODEL', model.xp)
    print('ENCODER', model.encoder.xp)
    print('RNN', model.rnn.xp)
    print('DECODER', model.decoder.xp)

    train_batch_size = 20
    eval_batch_size = 20
    num_steps = 20
    train_data = batchify(corpus.train, train_batch_size, model.xp)
    val_data = batchify(corpus.valid, eval_batch_size, model.xp)
    test_data = batchify(corpus.test, eval_batch_size, model.xp)

    ###############################################################################
    # Build the model
    ###############################################################################

    if args.save_params:
        np.savez(args.save_params, **model.save_parameters(prefix='Model/'))
        print('Saved parameters to', args.save_params)
        sys.exit()
    if args.load_params:
        model.load_parameters(dict(np.load(args.load_params)), prefix='Model/')
        print('Loaded parameters from', args.load_params)

    ###############################################################################
    # Training code
    ###############################################################################

    # criterion = nn.CrossEntropyLoss()
    criterion = partial(chainer_sequence_loss,
                        reduce_across_batch='mean',
                        reduce_across_timesteps='sum')

    # Loop over epochs.
    orig_lr = args.lr
    # best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, 13 + 1):
            lr_decay = 0.5 ** max(epoch - 4, 0.0)
            lr = orig_lr * lr_decay
            epoch_start_time = time.time()
            train(model, corpus, train_data, criterion, epoch,
                  lr, train_batch_size, num_steps, args.log_interval,
                  args.trace_data)
            val_loss = evaluate(model, corpus, val_data,
                                criterion, eval_batch_size, num_steps)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            # if not best_val_loss or val_loss < best_val_loss:
            #     with open(args.save, 'wb') as f:
            #         torch.save(model, f)
            #     best_val_loss = val_loss
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    # with open(args.save, 'rb') as f:
    #     model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(model, corpus, test_data,
                         criterion, eval_batch_size, num_steps)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


if __name__ == '__main__':
    main()
