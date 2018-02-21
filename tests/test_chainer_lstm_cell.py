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

import pytorch_lm.lstm_chainer as chi
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
            chi_cell = chi.LstmCell(self.input_size, self.hidden_size)
            chi_d = chi_cell.save_parameters(prefix='Model/')
            np.savez(weight_file, **chi_d)
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
        chi_cell = chi.LstmCell(input_size, hidden_size)
        if cuda:
            chi_cell.to_gpu()

        with tf.Graph().as_default() as graph:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope('Model', initializer=initializer):
                tfi_cell = tfi.LstmCell(input_size, hidden_size, batch_size)
            init = tf.global_variables_initializer()

        with tf.Session(graph=graph) as session:
            session.run(init)
            yield chi_cell, tfi_cell, session

    @unittest.skipUnless(chainer.cuda.available,
                         'CUDA test: a GPU is not available.')
    def test_cuda(self):
        """Tests if the chainer cell works on the GPU."""
        with self.__create_cells(
            self.input_size, self.hidden_size, self.batch_size, cuda=False
        ) as (chi_cell, tfi_cell, session):
            self.assertFalse(next(chi_cell.parameters()).is_cuda)
            chi_cell.load_parameters(chi_cell.save_parameters())
            self.assertFalse(next(chi_cell.parameters()).is_cuda)

        with self.__create_cells(
            self.input_size, self.hidden_size, self.batch_size, cuda=True
        ) as (chi_cell, tfi_cell, session):
            self.assertTrue(next(chi_cell.parameters()).is_cuda)
            chi_cell.load_parameters(chi_cell.save_parameters())
            self.assertTrue(next(chi_cell.parameters()).is_cuda)

    def __assert_cell_state_equals(self, chi_cell, tfi_cell, session):
        """Checks if the state vectors of the two cells are the same."""
        equals = []
        for name, _ in chi_cell.namedparams():
            chi_value = F.copy(getattr(chi_cell, name[1:]), -1).data
            tfi_value = session.run(getattr(tfi_cell, name[1:]))
            equals.append(np.allclose(chi_value, tfi_value))
        self.assertTrue(all(equals))
        # np.allclose(chi_value, tfi_d[name]for name, chi_value in chi_d.items()]

    def __test_data_from_chainer_to_tf(self, cuda):
        """Tests data transfer from the chainer cell to the tf one."""
        with self.__create_cells(
            self.input_size, self.hidden_size, self.batch_size, cuda=cuda
        ) as (chi_cell, tfi_cell, session):
            chi_d = chi_cell.save_parameters(prefix='Model/')
            tfi_cell.load_parameters(session, chi_d)

            self.__assert_cell_state_equals(chi_cell, tfi_cell, session)

    def test_data_from_chainer_to_tf(self):
        """Tests data transfer from the chainer cell to the tf one."""
        with self.subTest(name='cpu'):
            self.__test_data_from_chainer_to_tf(False)
        with self.subTest(name='gpu'):
            if not chainer.cuda.available:
                self.skipTest('CUDA test: a GPU is not available.')
            self.__test_data_from_chainer_to_tf(True)

    def __test_data_from_tf_to_chainer(self, cuda):
        """Tests data transfer from the tf cell to the chainer one."""
        with self.__create_cells(
            self.input_size, self.hidden_size, self.batch_size, cuda=cuda
        ) as (chi_cell, tfi_cell, session):
            tfi_d = tfi_cell.save_parameters(session)
            chi_cell.load_parameters(tfi_d, prefix='Model/')

            self.__assert_cell_state_equals(chi_cell, tfi_cell, session)

    def test_data_from_tf_to_chainer(self):
        """Tests data transfer from the tf cell to the chainer one."""
        with self.subTest(name='cpu'):
            self.__test_data_from_tf_to_chainer(False)
        with self.subTest(name='gpu'):
            if not chainer.cuda.available:
                self.skipTest('CUDA test: a GPU is not available.')
            self.__test_data_from_tf_to_chainer(True)

    def __run_loss_and_grad(self, chi_cell, tfi_cell, session, hidden_np, cuda):
            # Input
            input_np = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]],
                                dtype=np.float32)
            chi_input = Variable(input_np)
            if cuda:
                chi_input.to_gpu()
            tfi_input = tf.placeholder(tf.float32, [self.batch_size, self.input_size])

            # Initial states
            chi_hidden = chi_cell.init_hidden(np_arrays=hidden_np)
            # tfi_hidden = tfi_cell.init_hidden(np_arrays=hidden_np)
            tfi_init_state = tfi_cell.init_hidden()
            tfi_final_state = tfi_cell(tfi_input, tfi_init_state)

            # Chainer
            chi_cell.cleargrads()
            chi_h, chi_c = chi_cell(chi_input, chi_hidden)
            chi_target = F.broadcast_to(
                Variable(chi_cell.xp.array([0, 1], dtype=chi_cell.xp.float32)),
                shape=chi_h.shape
            )
            chi_loss = F.sqrt(F.sum((chi_h - chi_target) ** 2))
            chi_loss.backward()  # retain_graph=True)
            chi_h_np, chi_c_np = (F.copy(v, -1).data for v in (chi_h, chi_c))
            chi_loss_np = np.asscalar(F.copy(chi_loss, -1).data)
            chi_grads_dict = {name: F.copy(p.grad, -1).data
                              for name, p in chi_cell.namedparams()}

            # Tensorflow
            tfi_loss = tf.norm((tfi_final_state[0] - tf.constant([0, 1.0])))
            tfi_gradients = tf.gradients(tfi_loss, tfi_cell.weights)
            tfi_feed_dict = {tfi_input: input_np,
                             tfi_init_state: hidden_np}
            (tfi_h_np, tfi_c_np), tfi_loss_np, tfi_grads_list = session.run(
                [tfi_final_state, tfi_loss, tfi_gradients], tfi_feed_dict)
            tfi_grads_dict = {
                name: tfi_grads_list[tfi_cell.weights.index(
                    getattr(tfi_cell, name[1:]))]
                for name in chi_grads_dict
            }

            return ((chi_h_np, chi_c_np), (tfi_h_np, tfi_c_np),
                    chi_loss_np, tfi_loss_np, chi_grads_dict, tfi_grads_dict)

    def __run_one_step(self, hidden_np, cuda=False):
        """Runs one step from a user-specified hidden state."""
        with self.__create_cells(
            self.input_size, self.hidden_size, self.batch_size, cuda=cuda
        ) as (chi_cell, tfi_cell, session):
            tfi_cell.load_parameters(session, self.weights)
            chi_cell.load_parameters(self.weights, prefix='Model/')

            ((chi_h_np, chi_c_np), (tfi_h_np, tfi_c_np),
             chi_loss_np, tfi_loss_np, chi_grads_dict, tfi_grads_dict) = \
                self.__run_loss_and_grad(chi_cell, tfi_cell, session, hidden_np, cuda)

            # The tests
            with self.subTest(name='state_1'):
                self.assertTrue(np.allclose(tfi_h_np, chi_h_np))
                self.assertTrue(np.allclose(tfi_c_np, chi_c_np))

            with self.subTest(name='loss_1'):
                self.assertTrue(np.allclose(tfi_loss_np, chi_loss_np))

            with self.subTest(name='gradients_1'):
                for name, chi_value in chi_grads_dict.items():
                    self.assertTrue(np.allclose(chi_value, tfi_grads_dict[name]))

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
