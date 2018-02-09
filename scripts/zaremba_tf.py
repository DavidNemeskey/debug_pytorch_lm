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

import tensorflow as tf

from pytorch_lm.data import Corpus
from pytorch_lm.lstm_tf import Lstm


class SmallZarembaModel(object):
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

        self.rnn = Lstm(self.input_size, self.hidden_size,
                        self.num_layers, batch_size)
        self._initial_state = self.rnn.init_hidden()

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                'embedding', [vocab_size, self.hidden_size],
                trainable=True, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        outputs, state = self.rnn(inputs, self._initial_state)
        self._final_state = state

        softmax_w = tf.get_variable(
            "softmax_w", [self.hidden_size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable(
            "softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.einsum('ijk,kl->ijl', outputs, softmax_w) + softmax_b

        cost = tf.contrib.seq2seq.sequence_loss(
            logits,
            self._targets,
            tf.ones([batch_size, num_steps], dtype=tf.float32),
            average_across_timesteps=False,  # BPTT
            average_across_batch=True)
        self._cost = tf.reduce_sum(cost)
        self._prediction = tf.reshape(
            tf.nn.softmax(logits), [-1, num_steps, vocab_size])

        clip = 5
        if is_training:
            self._lr = tf.Variable(0.0, trainable=False)
            self.tvars = tf.trainable_variables()
            self.grads = tf.gradients(self.cost, self.tvars)
            if clip:
                self.clipped_grads, _ = tf.clip_by_global_norm(self.grads, clip)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(zip(self.clipped_grads, self.tvars))

            self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)
        else:
            self._train_op = tf.no_op()

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def predictions(self):
        return self._prediction

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

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


def train(sess, model, corpus, train_data, epoch, lr, batch_size,
          num_steps, log_interval):
    # Turn on training mode which enables dropout.
    model.assign_lr(sess, lr)
    total_loss = 0
    start_time = time.time()
    fetches = [model.cost, model.predictions, model.final_state, model.train_op]
    hidden = sess.run(model.initial_state)

    data_len = train_data.shape[1]
    for batch, i in enumerate(range(0, data_len - 1, num_steps)):
        # print('FOR', batch, i, (train_data.size(1) - 1) // num_steps)
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
        feed_dict = {
            model.input_data: data,
            model.targets: targets,
            tuple(model.initial_state): tuple(hidden)
        }
        cost, output, hidden, _, grads = sess.run(fetches + [model.grads], feed_dict)

        total_loss += cost / num_steps

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                  'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // num_steps, lr,
                      elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)),
                  flush=True)
            total_loss = 0
            start_time = time.time()


def evaluate(sess, model, corpus, data_source, batch_size, num_steps):
    total_loss = 0
    fetches = [model.cost, model.predictions, model.final_state]
    hidden = sess.run(model.initial_state)
    data_len = data_source.shape[1]

    for i in range(0, data_len - 1, num_steps):
        data, targets = get_batch(data_source, i, num_steps, evaluation=True)
        data = data.T
        targets = targets.T
        feed_dict = {
            model.input_data: data,
            model.targets: targets,
            tuple(model.initial_state): tuple(hidden)
        }
        cost, output, hidden = sess.run(fetches, feed_dict)
        total_loss += cost
    # print('TOTAL LOSS', total_loss, 'LEN DATA', len(data_source), data_source.size())
    return total_loss / len(data_source)


def main():
    args = parse_arguments()

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

    with tf.Graph().as_default() as graph:
        tf.set_random_seed(args.seed)
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.name_scope('Train'):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                mtrain = SmallZarembaModel(True, vocab_size, train_batch_size, num_steps)
        with tf.name_scope('Valid'):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = SmallZarembaModel(False, vocab_size, train_batch_size, num_steps)
        with tf.name_scope('Test'):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = SmallZarembaModel(False, vocab_size, train_batch_size, num_steps)
        with tf.name_scope('Global_ops'):
            init = tf.global_variables_initializer()

    ###############################################################################
    # Training code
    ###############################################################################

    # Loop over epochs.
    orig_lr = 1.0

    with tf.Session(graph=graph) as sess:
        sess.run(init)

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, 13 + 1):
                lr_decay = 0.5 ** max(epoch - 4, 0.0)
                lr = orig_lr * lr_decay
                epoch_start_time = time.time()
                train(sess, mtrain, corpus, train_data, epoch,
                      lr, train_batch_size, num_steps, args.log_interval)
                val_loss = evaluate(sess, mvalid, corpus, val_data,
                                    eval_batch_size, num_steps)
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
        test_loss = evaluate(sess, mtest, corpus, test_data,
                             eval_batch_size, num_steps)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)


if __name__ == '__main__':
    main()
