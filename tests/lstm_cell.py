import contextlib
import unittest

import numpy as np
import tensorflow as tf

import lstm_pytorch as pti
import lstm_tf as tfi


class TestLstmCells(unittest.TestCase):
    """Tests that the two LSTM cell implementations work alike."""
    @contextlib.contextmanager
    def __create_cells(self, input_size, hidden_size, batch_size):
        """
        Helper method. Creates three objects:
        - a pytorch LstmCell
        - a tensorflow LstmCell
        - a session for the latter.
        """
        pti_cell = pti.LstmCell(input_size, hidden_size)

        with tf.Graph().as_default() as graph:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope('Model', initializer=initializer):
                tfi_cell = tfi.LstmCell(input_size, hidden_size, batch_size)
            init = tf.global_variables_initializer()

        with tf.Session(graph=graph) as session:
            session.run(init)
            yield pti_cell, tfi_cell, session

    def test_data_from_pytorch_to_tf(self):
        """Tests data transfer from the pytorch cell to the tf one."""
        input_size, hidden_size = 3, 2
        batch_size = 4

        with tf.Graph().as_default() as graph:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope('Model', initializer=initializer):
                tfi_cell = tfi.LstmCell(input_size, hidden_size, batch_size)
            init = tf.global_variables_initializer()

        with tf.Session(graph=graph) as session:
            session.run(init)

            pti_cell = pti.LstmCell(input_size, hidden_size)
            pti_d = pti_cell.save_parameters(prefix='Model/')
            tfi_cell.load_parameters(session, pti_d)
            # tfi_d = tfi_cell.save_parameters(session)

            equals = []
            for name, _ in pti_cell.named_parameters():
                pti_value = getattr(pti_cell, name).data.cpu().numpy()
                tfi_value = session.run(getattr(tfi_cell, name))
                equals.append(np.allclose(pti_value, tfi_value))
            self.assertTrue(all(equals))
            # np.allclose(pti_value, tfi_d[name]for name, pti_value in pti_d.items()]

    def test_data_from_tf_to_pytorch(self):
        """Tests data transfer from the tf cell to the pytorch one."""
        input_size, hidden_size = 3, 2
        batch_size = 4

        with tf.Graph().as_default() as graph:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope('Model', initializer=initializer):
                tfi_cell = tfi.LstmCell(input_size, hidden_size, batch_size)
            init = tf.global_variables_initializer()

        with tf.Session(graph=graph) as session:
            session.run(init)

            pti_cell = pti.LstmCell(input_size, hidden_size)
            tfi_d = tfi_cell.save_parameters(session)
            pti_cell.load_parameters(tfi_d, prefix='Model/')

            equals = []
            for name, _ in pti_cell.named_parameters():
                pti_value = getattr(pti_cell, name).data.cpu().numpy()
                tfi_value = session.run(getattr(tfi_cell, name))
                equals.append(np.allclose(pti_value, tfi_value))
            self.assertTrue(all(equals))

    def test_output(self):
        """Tests whether the states are the same after reading an input."""
        pass


if __name__ == '__main__':
    unittest.main()
