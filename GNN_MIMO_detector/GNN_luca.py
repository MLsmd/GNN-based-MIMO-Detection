import torch as t

class GNN_fully_connected(t.nn.Module):
    def __init__(self, n, m, in_size, out_size,
                 size_vn_values, size_edge_embed, n_hidden_layers_prop, size_per_hidden_layer_prop, size_edge_values,
                 size_gru_hidden_state, size_agg_embed, size_out_layer_agg,
                 n_hidden_layers_readout, size_per_hidden_layer_readout,
                 device=t.device('cpu')):
        super(GNN_fully_connected, self).__init__()
        self.device = device
        self.n = n
        self.m = m
        # Initialize propagation, aggregation, and readout module.
        self.propagation = PropagationModule(size_vn_values=size_vn_values, size_edge_embed=size_edge_embed, n_hidden_layers=n_hidden_layers_prop, size_per_hidden_layer=size_per_hidden_layer_prop, size_edge_values=size_edge_values, device=device)
        self.aggregation = AggregationModule(size_gru_hidden_state=size_gru_hidden_state, size_edge_values=size_edge_values, size_agg_embed=size_agg_embed, size_out_layer=size_out_layer_agg, device=device)
        self.readout = ReadoutModule(size_vn_values=size_vn_values, out_size=out_size, n_hidden_layers=n_hidden_layers_readout, size_per_hidden_layer=size_per_hidden_layer_readout, device=device)

        # Linear layer to generate initial vn_values
        self.vn_value_init = t.nn.Linear(in_features=in_size, out_features=size_vn_values, device=device)

    def forward(self, init_features, edge_weight, noise_info, hx_init, cons, x_hat_MMSE, var_MMSE): #
        # Prepare transformation from luca model to github model
        batch_size = init_features.shape[0]
        n_gnn_iters = 10
        x = init_features
        edge_embed = t.zeros((batch_size, 2*self.n, 2*self.n, 2), device=self.device)
        count = 0
        for i in range(2*self.n):
            for j in range(2*self.m):
                if i != j:
                    edge_embed[:,i,j,0] = edge_weight[:,count]
                    count += 1
                    edge_embed[:,i,j,1] = noise_info[:,0]

        agg_embed = t.zeros(size=(batch_size,2*self.n,0), device=self.device)


        # Initialize vn values using 1 linear layer
        vn_val = self.vn_value_init(x)  # shape = (batch_size, n, size_vn_values)

        # Check some shapes.
        assert len(vn_val.shape) == 3
        batch_size = vn_val.shape[0]
        n_nodes = vn_val.shape[1]
        assert len(edge_embed.shape) == 4
        assert edge_embed.shape[:3] == (batch_size, n_nodes, n_nodes, )

        # Init gru state with correct size
        self.aggregation.init_gru_state(batch_size=batch_size, n=n_nodes, device=self.device)

        # Message passing
        for iter_l in range(n_gnn_iters):
            msg = self.propagation(vn_value_source=vn_val[:,:,None,:].repeat(1,1,n_nodes,1),
                                   vn_value_destination=vn_val[:,None,:,:].repeat(1,n_nodes,1,1),
                                   edge_embed=edge_embed)
            vn_val = self.aggregation(in_msgs=msg, agg_embed=agg_embed)

        # Readout: Returns refined cavity marginal distributions.
        return self.readout(vn_value=vn_val)

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
            x = self.linear_layers[i](x) # TODO no activation function?
        return self.linear_layers[-1](x) # no softmax to get log probabilities