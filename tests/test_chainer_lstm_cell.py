#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import contextlib
import importlib
import os
import unittest

try:
    chainer = importlib.import_module('chainer')
    import chainer.functions as F
    from chainer import Variable
except ImportError:
    chainer = None
import numpy as np
import tensorflow as tf

import pytorch_lm.lstm_chainer as pti
import pytorch_lm.lstm_tf as tfi


@unittest.skipIf(chainer is None, "Chainer not found.")
class TestChainerLstmCells(unittest.TestCase):
    """
    Tests that the Chainer LSTM cell implementation is the same as the TF one,
    too.
    """
    def setUp(self):
        self.input_size = 3
        self.hidden_size = 2
        self.batch_size = 4
        weight_file = os.path.join(os.path.dirname(__file__), 'lstm_cell.npz')
        state_file = os.path.join(os.path.dirname(__file__), 'lstm_state.npz')
        if not os.path.isfile(weight_file):
            pti_cell = pti.LstmCell(self.input_size, self.hidden_size)
            pti_d = pti_cell.save_parameters(prefix='Model/')
            np.savez(weight_file, **pti_d)
        if not os.path.isfile(state_file):
            h = np.random.rand(self.batch_size, self.hidden_size).astype(np.float32)
            c = np.random.rand(self.batch_size, self.hidden_size).astype(np.float32)
            np.savez(state_file, h=h, c=c)
        self.weights = dict(np.load(weight_file))
        states = dict(np.load(state_file))
        self.initial_hidden = tuple(states[k] for k in ['h', 'c'])

    @contextlib.contextmanager
    def __create_cells(self, input_size, hidden_size, batch_size, cuda=False):
        """
        Helper method. Creates three objects:
        - a chainer LstmCell
        - a tensorflow LstmCell
        - a session for the latter.
        """
        pti_cell = pti.LstmCell(input_size, hidden_size)
        if cuda:
            pti_cell.to_gpu()

        with tf.Graph().as_default() as graph:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope('Model', initializer=initializer):
                tfi_cell = tfi.LstmCell(input_size, hidden_size, batch_size)
            init = tf.global_variables_initializer()

        with tf.Session(graph=graph) as session:
            session.run(init)
            yield pti_cell, tfi_cell, session

    @unittest.skipUnless(chainer.cuda.available,
                         'CUDA test: a GPU is not available.')
    def test_cuda(self):
        """Tests if the Pytorch cell works on the GPU."""
        with self.__create_cells(
            self.input_size, self.hidden_size, self.batch_size, cuda=False
        ) as (pti_cell, tfi_cell, session):
            self.assertFalse(next(pti_cell.parameters()).is_cuda)
            pti_cell.load_parameters(pti_cell.save_parameters())
            self.assertFalse(next(pti_cell.parameters()).is_cuda)

        with self.__create_cells(
            self.input_size, self.hidden_size, self.batch_size, cuda=True
        ) as (pti_cell, tfi_cell, session):
            self.assertTrue(next(pti_cell.parameters()).is_cuda)
            pti_cell.load_parameters(pti_cell.save_parameters())
            self.assertTrue(next(pti_cell.parameters()).is_cuda)

    def __assert_cell_state_equals(self, pti_cell, tfi_cell, session):
        """Checks if the state vectors of the two cells are the same."""
        equals = []
        for name, _ in pti_cell.namedparams():
            pti_value = F.copy(getattr(pti_cell, name[1:]), -1).data
            tfi_value = session.run(getattr(tfi_cell, name[1:]))
            equals.append(np.allclose(pti_value, tfi_value))
        self.assertTrue(all(equals))
        # np.allclose(pti_value, tfi_d[name]for name, pti_value in pti_d.items()]

    def __test_data_from_pytorch_to_tf(self, cuda):
        """Tests data transfer from the pytorch cell to the tf one."""
        with self.__create_cells(
            self.input_size, self.hidden_size, self.batch_size, cuda=cuda
        ) as (pti_cell, tfi_cell, session):
            pti_d = pti_cell.save_parameters(prefix='Model/')
            tfi_cell.load_parameters(session, pti_d)

            self.__assert_cell_state_equals(pti_cell, tfi_cell, session)

    def test_data_from_pytorch_to_tf(self):
        """Tests data transfer from the pytorch cell to the tf one."""
        with self.subTest(name='cpu'):
            self.__test_data_from_pytorch_to_tf(False)
        with self.subTest(name='gpu'):
            if not chainer.cuda.available:
                self.skipTest('CUDA test: a GPU is not available.')
            self.__test_data_from_pytorch_to_tf(True)

    def __test_data_from_tf_to_pytorch(self, cuda):
        """Tests data transfer from the tf cell to the pytorch one."""
        with self.__create_cells(
            self.input_size, self.hidden_size, self.batch_size, cuda=cuda
        ) as (pti_cell, tfi_cell, session):
            tfi_d = tfi_cell.save_parameters(session)
            pti_cell.load_parameters(tfi_d, prefix='Model/')

            self.__assert_cell_state_equals(pti_cell, tfi_cell, session)

    def test_data_from_tf_to_pytorch(self):
        """Tests data transfer from the tf cell to the pytorch one."""
        with self.subTest(name='cpu'):
            self.__test_data_from_tf_to_pytorch(False)
        with self.subTest(name='gpu'):
            if not chainer.cuda.available:
                self.skipTest('CUDA test: a GPU is not available.')
            self.__test_data_from_tf_to_pytorch(True)

    def __run_loss_and_grad(self, pti_cell, tfi_cell, session, hidden_np, cuda):
            # Input
            input_np = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]],
                                dtype=np.float32)
            pti_input = Variable(input_np)
            if cuda:
                pti_input.to_gpu()
            tfi_input = tf.placeholder(tf.float32, [self.batch_size, self.input_size])

            # Initial states
            pti_hidden = pti_cell.init_hidden(np_arrays=hidden_np)
            # tfi_hidden = tfi_cell.init_hidden(np_arrays=hidden_np)
            tfi_init_state = tfi_cell.init_hidden()
            tfi_final_state = tfi_cell(tfi_input, tfi_init_state)

            # Chainer
            pti_cell.cleargrads()
            pti_h, pti_c = pti_cell(pti_input, pti_hidden)
            pti_target = Variable(pti_cell.xp.array([0, 1], dtype=xp.float32))
            pti_loss = (pti_h - pti_target).norm(2)
            pti_loss.backward()  # retain_graph=True)
            pti_h_np, pti_c_np = (F.copy(v, -1).data for v in (pti_h, pti_c))
            pti_loss_np = F.copy(pti_loss, -1).data[0]
            pti_grads_dict = {name: F.copy(p.grad, -1).data
                              for name, p in pti_cell.namedparams()}

            # Tensorflow
            tfi_loss = tf.norm((tfi_final_state[0] - tf.constant([0, 1.0])))
            tfi_gradients = tf.gradients(tfi_loss, tfi_cell.weights)
            tfi_feed_dict = {tfi_input: input_np,
                             tfi_init_state: hidden_np}
            (tfi_h_np, tfi_c_np), tfi_loss_np, tfi_grads_list = session.run(
                [tfi_final_state, tfi_loss, tfi_gradients], tfi_feed_dict)
            tfi_grads_dict = {
                name: tfi_grads_list[tfi_cell.weights.index(getattr(tfi_cell, name))]
                for name in pti_grads_dict
            }

            return ((pti_h_np, pti_c_np), (tfi_h_np, tfi_c_np),
                    pti_loss_np, tfi_loss_np, pti_grads_dict, tfi_grads_dict)

    def __run_one_step(self, hidden_np, cuda=False):
        """Runs one step from a user-specified hidden state."""
        with self.__create_cells(
            self.input_size, self.hidden_size, self.batch_size, cuda=cuda
        ) as (pti_cell, tfi_cell, session):
            tfi_cell.load_parameters(session, self.weights)
            pti_cell.load_parameters(self.weights, prefix='Model/')

            ((pti_h_np, pti_c_np), (tfi_h_np, tfi_c_np),
             pti_loss_np, tfi_loss_np, pti_grads_dict, tfi_grads_dict) = \
                self.__run_loss_and_grad(pti_cell, tfi_cell, session, hidden_np, cuda)

            # The tests
            with self.subTest(name='state_1'):
                self.assertTrue(np.allclose(tfi_h_np, pti_h_np))
                self.assertTrue(np.allclose(tfi_c_np, pti_c_np))

            with self.subTest(name='loss_1'):
                self.assertTrue(np.allclose(tfi_loss_np, pti_loss_np))

            with self.subTest(name='gradients_1'):
                for name, pti_value in pti_grads_dict.items():
                    self.assertTrue(np.allclose(pti_value, tfi_grads_dict[name]))

    def test_from_zero(self):
        """
        Runs one round of input through both cells, and checks
        - if the resulting state vectors are the same
        - if the loss w.r.t. a dummy target are the same
        - if the gradients are the same
        - if, after having applied the gradients, the new weights are the same.

        This method starts from the zero initial state.
        """
        hidden_np = (
            np.zeros((self.batch_size, self.hidden_size), dtype=np.float32),
            np.zeros((self.batch_size, self.hidden_size), dtype=np.float32)
        )
        with self.subTest(name='cpu'):
            self.__run_one_step(hidden_np, False)
        with self.subTest(name='gpu'):
            if not chainer.cuda.available:
                self.skipTest('CUDA test: a GPU is not available.')
            self.__run_one_step(hidden_np, True)

    def test_from_state(self):
        """
        Same as test_from_zero(), but starts from a randomly generated
        hidden state.
        """
        with self.subTest(name='cpu'):
            self.__run_one_step(self.initial_hidden, False)
        with self.subTest(name='gpu'):
            if not chainer.cuda.available:
                self.skipTest('CUDA test: a GPU is not available.')
            self.__run_one_step(self.initial_hidden, True)


if __name__ == '__main__':
    unittest.main()
