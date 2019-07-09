from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq

from lib.utils import load_graph_data
from lib.metrics import masked_mae_loss

from model.dcrnn_cell import DCGRUCell


class DCRNNModel(object):
    def __init__(self, is_training, batch_size, scaler, adj_matrix_file, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mae = None
        self._train_op = None

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 0))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        filter_type = model_kwargs.get('filter_type', 'laplacian')

        networkType = model_kwargs.get('network', 'gconv') # fc/gconv
        matrixType = model_kwargs.get('weightMatrix') # a/d
        attention = model_kwargs.get('attention')

        horizon = int(model_kwargs.get('horizon', 1))
        max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))
        aux_dim = input_dim - output_dim

        _, _, adj_mx = load_graph_data(adj_matrix_file)

        graphEmbedFile = None
        if networkType == 'fc':
            graphEmbedFile = model_kwargs.get('graphEmbedFile')
        # input_dim = 2
        # output_dim = 1
        # Input (batch_size, timesteps, num_sensor, input_dim)
        # print(batch_size, seq_len, num_nodes, input_dim)
        # 64 12 207 2
        # Batch size is a term used in machine learning and refers to the number of training examples utilised in one iteration.
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')

        # GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * input_dim))
        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * output_dim))

        cell = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes, 
                         network_type=networkType, graphEmbedFile=graphEmbedFile, filter_type=filter_type)
        cell_with_projection = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                         network_type=networkType, graphEmbedFile=graphEmbedFile, num_proj=output_dim, filter_type=filter_type)
        encoding_cells = [cell] * num_rnn_layers
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        # projection is for the last step of decoding
        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)
        # print('We have initiated the cells.')

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            # What are the inputs and labels??

            # labels are ground truth

            # What is input_dim and output_dim
            # input_dim = 2
            # output_dim = 1
            inputs = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            labels = tf.unstack(
                tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1)
            if aux_dim > 0:
                aux_info = tf.unstack(self._labels[..., output_dim:], axis=1)
                aux_info.insert(0, None)
            labels.insert(0, GO_SYMBOL)
            # print('Did we arrive here? Yes we did.')

            def _loop_function(prev, i):
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if use_curriculum_learning:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: labels[i], lambda: prev)
                    else:
                        result = labels[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                # print(result.shape)
                # exit()
                # (64, 207)
                if False and aux_dim > 0:
                    result = tf.reshape(result, (batch_size, num_nodes, output_dim))
                    # print(result.shape)
                    # (64, 207, 1)
                    result = tf.concat([result, aux_info[i]], axis=-1)
                    # print(result.shape)
                    # (64, 207, 2)
                    result = tf.reshape(result, (batch_size, num_nodes * input_dim))
                    # print(result.shape)
                    # print(result.shape)
                    # (64, 414)
                return result

            # tf.contrib.rnn.static_rnn: https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/contrib/rnn/static_rnn
            # Creates a recurrent neural network specified by RNNCell: cell.
            # _gconv is called several times in this step
            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            # exit()
            # ****** HaHa ****** appeared 24 times
            # exit()
            # outputs is a list
            # Inside the decoder function, there is a loop function that probably propogates in the rnn structure
            # there are many printouts for calling the cells as a function, in the _gconv 

            # outputs is of 13 such rnn cells
            # <tf.Tensor 'Train/DCRNN/DCRNN_SEQ/rnn_decoder/rnn_decoder/multi_rnn_cell/cell_1_12/dcgru_cell/projection/Reshape_1:0' shape=(64, 207) dtype=float32>

            # final_state is of 2 such rnn cells
            # <tf.Tensor 'Train/DCRNN/DCRNN_SEQ/rnn_decoder/rnn_decoder/multi_rnn_cell/cell_0_12/dcgru_cell/add:0' shape=(64, 13248) dtype=float32>
            # print('We are now in decoding')
            # tf.contrib.legacy_seq2seq.rnn_decoder: https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/rnn_decoder
            # RNN decoder for the sequence-to-sequence model.
            # _gconv is called several times in this step
            outputs, final_state = legacy_seq2seq.rnn_decoder(labels, enc_state, decoding_cells,
                                                              loop_function=_loop_function)

        # print("Did we arrive here? No we didn't.")
        # Project the output to output_dim.
        # https://www.tensorflow.org/api_docs/python/tf/stack
        # Why remove the last element?
        outputs = tf.stack(outputs[:-1], axis=1)
        # outputs is not a list anymore, but a stacked tensor
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')
        self._merged = tf.summary.merge_all()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs
