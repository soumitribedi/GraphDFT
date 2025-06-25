import torch
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool as gap, global_max_pool as gmp

class NN_Simple(torch.nn.Module):
    def __init__(self, num_features, embedding_size=32, num_layers=5):
        """
        Initialize the GNN model with a specified number of layers.

        Parameters:
        ----------
        num_features : int
            The number of input features for each node in the graph.

        embedding_size : int, optional
            The size of the embedding (hidden) layer. Default is 32.

        num_layers : int, optional
            The number of Linear layers in the model. Default is 3, require >=2.


        """
        super(NN_Simple, self).__init__()
        torch.manual_seed(42)

        self.num_layers = num_layers
        self.num_features = num_features
        self.embedding_size = embedding_size

        ## NN_Simple
        self.input_layer = Linear(num_features, embedding_size)
        self.hidden_layers = torch.nn.ModuleList([Linear(embedding_size, embedding_size) for _ in range(num_layers)])
        self.output_layer = Linear(embedding_size, 1)

        self.act = ReLU()


    def forward(self, x):
        """
        Perform a forward pass through the GNN model.

        Parameters:
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, num_features].

        Returns:
        -------
        out : torch.Tensor
            The output predictions of shape [num_nodes, output_dim], where output_dim is typically 1.

        hidden : torch.Tensor
            The final hidden node embeddings of shape [num_nodes, embedding_size], capturing learned node representations.
        """
        # input layer
        hidden = self.input_layer(x)
        hidden = self.act(hidden)

        # hidden layers
        for hidden_layer in self.hidden_layers:
            hidden = hidden_layer(hidden)
            hidden = self.act(hidden)

        # Apply a final (linear) classifier.
        out = self.output_layer(hidden)

        return out, hidden