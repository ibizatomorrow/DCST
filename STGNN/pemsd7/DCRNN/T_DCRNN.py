import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import random
from STGNN.pemsd7.DCRNN.util import *

class DiffusionGraphConv(nn.Module):
    def __init__(self, supports, input_dim, hid_dim, num_nodes, max_diffusion_step, output_dim, bias_start=0.0):
        super(DiffusionGraphConv, self).__init__()
        self.num_matrices = len(supports) * max_diffusion_step + 1  # Don't forget to add for x itself.
        input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports
        self.weight = nn.Parameter(torch.FloatTensor(size=(input_size*self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, state, output_size, bias_start=0.0):
        """
        Diffusion Graph convolution with graph matrix
        :param inputs:
        :param state:
        :param output_size:
        :param bias_start:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]
        # dtype = inputs.dtype

        x = inputs_and_state
        x0 = torch.transpose(x, dim0=0, dim1=1)
        x0 = torch.transpose(x0, dim0=1, dim1=2)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, self._num_nodes, input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self.num_matrices])

        x = torch.matmul(x, self.weight.cuda())  # (batch_size * self._num_nodes, output_size)
        x = torch.add(x, self.biases.cuda())
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """
    def __init__(self, input_dim, num_units, adj_mat, max_diffusion_step, num_nodes,
                 num_proj=None, activation=torch.tanh, use_gc_for_ru=True, filter_type='laplacian'):
        """
        :param num_units: the hidden dim of rnn
        :param adj_mat: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param max_diffusion_step: the max diffusion step
        :param num_nodes:
        :param num_proj: num of output dim, defaults to 1 (speed)
        :param activation: if None, don't do activation for cell state
        :param use_gc_for_ru: decide whether to use graph convolution inside rnn
        """
        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._use_gc_for_ru = use_gc_for_ru
        self._supports = []
        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mat, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mat).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mat))
            supports.append(calculate_random_walk_matrix(adj_mat.T))
        else:
            supports.append(calculate_scaled_laplacian(adj_mat))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support).cuda())  # to PyTorch sparse tensor
        # supports = utils.calculate_scaled_laplacian(adj_mat, lambda_max=None)  # scipy coo matrix
        # self._supports = self._build_sparse_matrix(supports).cuda()  # to pytorch sparse tensor

        self.dconv_gate = DiffusionGraphConv(supports=self._supports, input_dim=input_dim,
                                             hid_dim=num_units, num_nodes=num_nodes,
                                             max_diffusion_step=max_diffusion_step,
                                             output_dim=num_units*2)
        self.dconv_candidate = DiffusionGraphConv(supports=self._supports, input_dim=input_dim,
                                                  hid_dim=num_units, num_nodes=num_nodes,
                                                  max_diffusion_step=max_diffusion_step,
                                                  output_dim=num_units)
        if num_proj is not None:
            self.project = nn.Linear(self._num_units, self._num_proj)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def forward(self, inputs, state):
        """
        :param inputs: (B, num_nodes * input_dim)
        :param state: (B, num_nodes * num_units)
        :return:
        """
        output_size = 2 * self._num_units
        # we start with bias 1.0 to not reset and not update
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(value, split_size_or_sections=int(output_size/2), dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))
        c = self.dconv_candidate(inputs, r * state, self._num_units)  # batch_size, self._num_nodes * output_size
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c
        if self._num_proj is not None:
            # apply linear projection to state
            batch_size = inputs.shape[0]
            output = torch.reshape(new_state, shape=(-1, self._num_units))  # (batch*num_nodes, num_units)
            output = torch.reshape(self.project(output), shape=(batch_size, self.output_size))  # (50, 207*1)
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        :return:
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        pass

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self._num_nodes * self._num_units)


class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, adj_mat, max_diffusion_step, hid_dim, num_nodes,
                 num_rnn_layers, filter_type):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self._num_rnn_layers = num_rnn_layers

        # encoding_cells = []
        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(DCGRUCell(input_dim=input_dim, num_units=hid_dim, adj_mat=adj_mat,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes, filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(input_dim=hid_dim, num_units=hid_dim, adj_mat=adj_mat,
                                            max_diffusion_step=max_diffusion_step,
                                            num_nodes=num_nodes, filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state):
        # inputs shape is (seq_length, batch, num_nodes, input_dim) (12, 64, 207, 2)
        # inputs to cell is (batch, num_nodes * input_dim)
        # init_hidden_state should be (num_layers, batch_size, num_nodes*num_units) (2, 64, 207*64)
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))  # (12, 64, 207*2)

        current_inputs = inputs
        output_hidden = []  # the output hidden states, shape (num_layers, batch, outdim)
        for i_layer in range(self._num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](current_inputs[t, ...], hidden_state)  # (50, 207*64)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).cuda()  # seq_len, B, ...
        # output_hidden: the hidden state of each layer at last time step, shape (num_layers, batch, outdim)
        # current_inputs: the hidden state of the top layer (seq_len, B, outdim)
        return output_hidden, current_inputs
    
    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in range(self._num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # init_states shape (num_layers, batch_size, num_nodes*num_units)
        return torch.stack(init_states, dim=0)

class DCGRUDecoder(nn.Module):
    def __init__(self, input_dim, adj_mat, max_diffusion_step, num_nodes,
                 hid_dim, output_dim, num_rnn_layers, filter_type):
        super(DCGRUDecoder, self).__init__()
        self.hid_dim = hid_dim
        self._num_nodes = num_nodes  # 207
        self._output_dim = output_dim  # should be 1
        self._num_rnn_layers = num_rnn_layers

        cell = DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
                         adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
                         num_nodes=num_nodes, filter_type=filter_type)
        cell_with_projection = DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
                                         adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
                                         num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)

        decoding_cells = list()
        # first layer of the decoder
        decoding_cells.append(DCGRUCell(input_dim=input_dim, num_units=hid_dim,
                                        adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes, filter_type=filter_type))
        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers - 1):
            decoding_cells.append(cell)
        decoding_cells.append(cell_with_projection)
        self.decoding_cells = nn.ModuleList(decoding_cells)

    def forward(self, inputs, initial_hidden_state, teacher_forcing_ratio=0.5):
        """
        :param inputs: shape should be (seq_length+1, batch_size, num_nodes, input_dim)
        :param initial_hidden_state: the last hidden state of the encoder. (num_layers, batch, outdim)
        :param teacher_forcing_ratio:
        :return: outputs. (seq_length, batch_size, num_nodes*output_dim) (12, 50, 207*1)
        """
        # inputs shape is (seq_length, batch, num_nodes, input_dim) (12, 50, 207, 1)
        # inputs to cell is (batch, num_nodes * input_dim)
        seq_length = inputs.shape[0]  # should be 13
        batch_size = inputs.shape[1]
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))  # (12+1, 50, 207*1)

        # tensor to store decoder outputs
        outputs = torch.zeros(seq_length, batch_size, self._num_nodes*self._output_dim)  # (13, 50, 207*1)
        # if rnn has only one layer
        # if self._num_rnn_layers == 1:
        #     # first input to the decoder is the GO Symbol
        #     current_inputs = inputs[0]  # (64, 207*1)
        #     hidden_state = prev_hidden_state[0]
        #     for t in range(1, seq_length):
        #         output, hidden_state = self.decoding_cells[0](current_inputs, hidden_state)
        #         outputs[t] = output  # (64, 207*1)
        #         teacher_force = random.random() < teacher_forcing_ratio
        #         current_inputs = (inputs[t] if teacher_force else output)

        current_input = inputs[0]  # the first input to the rnn is GO Symbol
        for t in range(1, seq_length):
            # hidden_state = initial_hidden_state[i_layer]  # i_layer=0, 1, ...
            next_input_hidden_state = []
            for i_layer in range(0, self._num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](current_input, hidden_state)
                current_input = output  # the input of present layer is the output of last layer
                next_input_hidden_state.append(hidden_state)  # store each layer's hidden state
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)
            outputs[t] = output  # store the last layer's output to outputs tensor
            # perform scheduled sampling teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio  # a bool value
            current_input = (inputs[t] if teacher_force else output)

        return outputs

class DCRNNModel(nn.Module):
    def __init__(self, adj_mat, batch_size, enc_input_dim, dec_input_dim, max_diffusion_step, num_nodes,
                 num_rnn_layers, rnn_units, seq_len, output_dim, filter_type):
        super(DCRNNModel, self).__init__()
        # scaler for data normalization
        # self._scaler = scaler
        self._batch_size = batch_size

        # max_grad_norm parameter is actually defined in data_kwargs
        self._num_nodes = num_nodes  # should be 207
        self._num_rnn_layers = num_rnn_layers  # should be 2
        self._rnn_units = rnn_units  # should be 64
        self._seq_len = seq_len  # should be 12
        # use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))  # should be true
        self._output_dim = output_dim  # should be 1

        # specify a GO symbol as the start of the decoder
        self.GO_Symbol = torch.zeros(1, batch_size, num_nodes * self._output_dim, 1).cuda()

        self.encoder = DCRNNEncoder(input_dim=enc_input_dim, adj_mat=adj_mat,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers, filter_type=filter_type)
        self.decoder = DCGRUDecoder(input_dim=dec_input_dim,
                                    adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
                                    num_nodes=num_nodes, hid_dim=rnn_units,
                                    output_dim=self._output_dim,
                                    num_rnn_layers=num_rnn_layers, filter_type=filter_type)
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
    def forward(self, source, target, teacher_forcing_ratio):
         # the size of source/target would be (64, 12, 207, 2)
        source = torch.transpose(source, dim0=0, dim1=1)
        target = torch.transpose(target[..., :self._output_dim], dim0=0, dim1=1)
        target = torch.cat([self.GO_Symbol, target], dim=0)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(self._batch_size).cuda()

        # last hidden state of the encoder is the context
        context, _ = self.encoder(source, init_hidden_state)  # (num_layers, batch, outdim)

        outputs = self.decoder(target, context, teacher_forcing_ratio=teacher_forcing_ratio)
        # the elements of the first time step of the outputs are all zeros.
        return outputs[1:, :, :]  # (seq_length, batch_size, num_nodes*output_dim)  (12, 64, 207*1)