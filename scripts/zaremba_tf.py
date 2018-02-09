#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Implements the small model from Zaremba (2014). Its main purpose is to
serve as a sanity check, because a TF implementation can perfectly reproduce
Zaremba's results (see emLam).
"""

import argparse
import math
import time

from pytorch_lm.data import Corpus
from pytorch_lm.lstm_tf import Lstm


class SmallZarembaModel(nn.Module):
    """"Implements the small model from Zaremba (2014)."""
    def __init__(self, is_training, vocab_size, batch_size, num_steps):
        super(SmallZarembaModel, self).__init__()
        self.hidden_size = 200
        self.input_size = 200
        self.num_layers = 2

        dims = [batch_size, num_steps]
        self._input_data = tf.placeholder(
            tf.int32, dims, name='input_placeholder')
        self._targets = tf.placeholder(
            tf.int32, dims, name='target_placeholder')

        self.cell = Lstm(self.input_size, self.hidden_size,
                         self.num_layers, batch_size)
        self._initial_state = cell.init_hidden()

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                'embedding', [vocab_size, self.hidden_Size],
                trainable=True, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        self.encoder = nn.Embedding(vocab_size, self.hidden_size)
        self.rnn = Lstm(self.input_size, self.hidden_size, self.num_layers)
        self.decoder = nn.Linear(self.hidden_size, vocab_size)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.uniform_(-initrange, initrange)  # fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        # print('INPUT', input.size())
        emb = self.encoder(input)
        # print('EMB', emb.size())
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
                    'described, in TensorFlow.')
    parser.add_argument('--data', '-d', type=str, default='./data/wikitext-2',
                        help='location of the data corpus (files called '
                             'train|valid|test.txt).')
    parser.add_argument('--model', '-m', type=str, default='LSTM',
                        help='the model key name.')
    parser.add_argument('--seed', '-s', type=int, default=1111, help='random seed')
    parser.add_argument('--log-interval', '-l', type=int, default=200, metavar='N',
                        help='report interval')
    return parser.parse_args()


def batchify(data, bsz):
    """Same as the PT function, only we stay in numpy all along."""
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    # data = data.view(bsz, -1).t().contiguous()
    data = data.reshape(bsz, -1)
    return data


def get_batch(source, i, num_steps, evaluation=False):
    """Same as the PT function, only we stay in numpy all along."""
    seq_len = min(num_steps, source.shape[1] - 1 - i)
    # TODO can we no_grad target as well?
    data = source[:, i:i+seq_len]
    target = source[:, i+1:i+1+seq_len]
    return data, target


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = Corpus(args.data)

    train_batch_size = 20
    eval_batch_size = 20
    num_steps = 20
    train_data = batchify(corpus.train, train_batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################

    vocab_size = len(corpus.dictionary)
    model = SmallZarembaModel(vocab_size)

    ###############################################################################
    # Training code
    ###############################################################################

    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    orig_lr = 1.0
    # best_val_loss = None

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
