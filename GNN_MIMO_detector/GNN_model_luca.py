import torch as t
from time_distributed import TimeDistributed,TimeDistributed_GRU

class GNN(t.nn.Module):
    def __init__(self, user_num,
                 in_size, out_size,
                 size_vn_values, size_edge_embed, n_hidden_layers_prop, size_per_hidden_layer_prop, size_edge_values,
                 size_gru_hidden_state, size_agg_embed, size_out_layer_agg,
                 n_hidden_layers_readout, size_per_hidden_layer_readout,
                 device=t.device('cpu'),
                 n_gnn_iters=10):
        super(GNN, self).__init__()
        self.device = device
        self.num_iter = n_gnn_iters
        self.user_num = user_num

        # Initialize propagation, aggregation, and readout module.
        self.propagation = PropagationModule(size_vn_values=size_vn_values, size_edge_embed=size_edge_embed, n_hidden_layers=n_hidden_layers_prop, size_per_hidden_layer=size_per_hidden_layer_prop, size_edge_values=size_edge_values, device=device)
        self.aggregation = AggregationModule(size_gru_hidden_state=size_gru_hidden_state, size_edge_values=size_edge_values, size_agg_embed=size_agg_embed, size_out_layer=size_out_layer_agg, device=device)
        self.readout = ReadoutModule(size_vn_values=size_vn_values, out_size=out_size, n_hidden_layers=n_hidden_layers_readout, size_per_hidden_layer=size_per_hidden_layer_readout, device=device)
        self.gru = TimeDistributed_GRU(t.nn.GRUCell, size_edge_values, size_gru_hidden_state)

        # Linear layer to generate initial vn_values
        self.vn_value_init = t.nn.Linear(in_features=in_size, out_features=size_vn_values, device=device)

    def forward(self, init_features, edge_weight, noise_info, hx_init, cons, x_hat_MMSE, var_MMSE):
        '''===========================custom layers========================'''

        def NN_iterations(initfeats, idx_iter, gru_read, hx):

            if idx_iter == 0:
                x_init = initfeats
                init_nodes = self.vn_value_init(x_init)  # sample, nodes, features
            else:
                init_nodes = gru_read

            # Interferance probability feature calculation
            slicing1 = init_nodes[:, temp_a, :]
            slicing2 = init_nodes[:, temp_b, :]

            edge_slicing = edge_weight[:, :, None]
            noise_slicing = noise_info.repeat(1, len(temp_a))[:, :, None]
            combined_slicing = t.cat((slicing1, slicing2, edge_slicing, noise_slicing), 2)

            messages_from_i_to_j = self.propagation.act_fct(self.propagation.linear_layers[0](combined_slicing))
            # messages_from_i_to_j= self.dropout1(messages_from_i_to_j)
            messages_from_i_to_j = self.propagation.act_fct(self.propagation.linear_layers[1](messages_from_i_to_j))
            # messages_from_i_to_j= self.dropout1(messages_from_i_to_j)
            messages_from_i_to_j = self.propagation.act_fct(self.propagation.linear_layers[2](messages_from_i_to_j))

            # Interferance probability features summation
            step = self.user_num - 1

            sum_messages = []
            for i in range(0, messages_from_i_to_j.shape[1], step):
                sum_messages.append(t.unsqueeze(t.sum((messages_from_i_to_j[:, i:i + step, :]), 1), 1))
            sum_messages = t.cat(sum_messages, 1)

            # GRU and outputs
            gru_out = self.gru(sum_messages, hx)
            gru_read = self.aggregation.linear_layer(gru_out)

            # gru_out = self.gru_small(sum_messages,init_nodes)
            # gru_read = gru_out

            return gru_read, gru_out

        # slicing parameters
        # p_x_y_saved=[]
        temp_a = []
        temp_b = []
        for i in range(self.user_num):
            for j in range(self.user_num):
                if i != j:
                    temp_a.append(i)
                    temp_b.append(j)

        # Placeholders and Initializations
        hx = hx_init
        initfeats = init_features
        xhat_MMSE = x_hat_MMSE
        var_MMSE = var_MMSE
        gru_read = None

        for idx_iter in range(self.num_iter):
            gru_read, hx = NN_iterations(initfeats, idx_iter, gru_read, hx)

        # gru_out = self.f4(gru_out)
        R_out1 = self.readout.linear_layers[0](gru_read)
        # R_out1= self.dropout2(R_out1)
        R_out2 = self.readout.linear_layers[1](R_out1)
        # R_out2= self.dropout2(R_out2)
        z = self.readout.linear_layers[2](R_out2)

        # z_reshaped = z.contiguous().view(-1,z.size(-1))  # (samples * nodes, features)
        # z_normalized = self.softmax(z_reshaped)
        # z_normalized = z_reshaped

        # p_x_y = z_normalized.contiguous().view(-1, z.size(1), z.size(-1))

        #     p_x_y_saved.append(p_x_y)
        # p_x_y_final =  torch.cat(p_x_y_saved,0)
        return z

class PropagationModule(t.nn.Module):
    def __init__(self, size_vn_values, size_edge_embed, n_hidden_layers, size_per_hidden_layer, size_edge_values,
                 device=t.device('cpu')):
        super(PropagationModule, self).__init__()
        assert n_hidden_layers >= 1
        self.n_hidden_layers = n_hidden_layers

        # Setup Linear layers.
        self.linear_layers = [t.nn.Linear(in_features=size_edge_embed + 2 * size_vn_values, out_features=size_per_hidden_layer[0], device=device), ] # Input layer
        for i in range(n_hidden_layers-1): # Hidden layers
            self.linear_layers.append( t.nn.Linear(in_features=size_per_hidden_layer[i], out_features=size_per_hidden_layer[i+1], device=device) )
        self.linear_layers.append(t.nn.Linear(in_features=size_per_hidden_layer[-1], out_features=size_edge_values, device=device)) # Output layer
        # Setup activation functions -> always ReLU.
        self.act_fct = t.nn.ReLU()
        # TODO insert dropout layers (in all modules)?

    def forward(self, vn_value_source, vn_value_destination, edge_embed):
        x = t.cat((vn_value_destination, vn_value_source, edge_embed), dim=-1) # Collect inputs for each edge.
        for i in range(self.n_hidden_layers):
            x = self.act_fct( self.linear_layers[i](x) )
        return self.linear_layers[-1](x)

class AggregationModule(t.nn.Module):
    def __init__(self, size_gru_hidden_state, size_edge_values, size_agg_embed, size_out_layer,
                 device=t.device('cpu')):
        super(AggregationModule, self).__init__()
        self.size_gru_hidden_state = size_gru_hidden_state
        # Gated recurrent unit (GRU)
        self.gru = t.nn.GRUCell(input_size=size_edge_values + size_agg_embed, hidden_size=size_gru_hidden_state, device=device)
        # Linear Layer
        self.linear_layer = t.nn.Linear(in_features=size_gru_hidden_state, out_features=size_out_layer, device=device)

    def init_gru_state(self, batch_size, n, device):
        self.gru_hidden_state = t.zeros(size=(batch_size, n, self.size_gru_hidden_state), device=device)

    def forward(self, in_msgs, agg_embed):
        assert len(in_msgs.shape) == 4 # shape = (batch_size, n FNs, n VNs, msg_size)
        x = t.cat((t.sum(in_msgs, dim=1) - t.diagonal(in_msgs, dim1=1,dim2=2).transpose(1,2), agg_embed), dim=-1) # sum is over all incoming msgs to each VN, i.e., over the FN dimension
        self.gru_hidden_state = self.gru(x.flatten(0,-2), self.gru_hidden_state.flatten(0,-2)).view(self.gru_hidden_state.shape)
        return self.linear_layer(self.gru_hidden_state)

class ReadoutModule(t.nn.Module):
    def __init__(self, size_vn_values, out_size, n_hidden_layers, size_per_hidden_layer, device=t.device('cpu')):
        super(ReadoutModule, self).__init__()
        assert n_hidden_layers >= 1
        self.n_hidden_layers = n_hidden_layers

        # Setup Linear layers.
        self.linear_layers = [t.nn.Linear(in_features=size_vn_values, out_features=size_per_hidden_layer[0], device=device), ] # Input layer
        for i in range(n_hidden_layers-1): # Hidden layers
            self.linear_layers.append( t.nn.Linear(in_features=size_per_hidden_layer[i], out_features=size_per_hidden_layer[i+1], device=device) )
        self.linear_layers.append(t.nn.Linear(in_features=size_per_hidden_layer[-1], out_features=out_size, device=device)) # Output layer
        # Setup activation functions -> always ReLU.
        self.act_fct = t.nn.ReLU()
        # Softmax
        self.softmax = t.nn.Softmax(dim=-1)

    def forward(self, vn_value):
        x = vn_value.clone()
        for i in range(self.n_hidden_layers):
            x = self.act_fct( self.linear_layers[i](x) )
        return self.softmax(self.linear_layers[-1](x))