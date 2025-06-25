import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GCN(torch.nn.Module):
    """
    A Graph Convolutional Network (GCN) to predict a molecular property from node features.

    This class implements a multi-layer GCN model using PyTorch Geometric. It is designed
    to process graph data by stacking several graph convolutional layers followed by a
    global pooling layer and a linear classifier.

    Attributes:
    ----------
    initial_conv : GCNConv
        The initial graph convolutional layer that transforms input node features to
        an embedding space.

    conv1 : GCNConv
        The first hidden graph convolutional layer.

    conv2 : GCNConv
        The second hidden graph convolutional layer.

    conv3 : GCNConv
        The third hidden graph convolutional layer.

    out : Linear
        A fully connected linear layer that maps the final hidden representations
        to the output space, typically used for classification.

    Methods:
    -------
    forward(x, edge_index, batch_index):
        Forward pass of the GCN model. Takes input node features, edge indices, and
        batch indices, and returns the output predictions and hidden node embeddings.

    Example:
    -------
    >>> gcn = GCN()
    >>> out, hidden = gcn(x, edge_index, batch_index)

    """
    def __init__(self,num_features,embedding_size=32,num_layers=3):
        """
        Initialize the GCN model with a specified number of layers.

        Parameters:
        ----------
        num_features : int
            The number of input features for each node in the graph.

        embedding_size : int, optional
            The size of the embedding (hidden) layer. Default is 32.

        num_layers : int, optional
            The number of graph convolutional layers in the model. Default is 3.

        """
        super(GCN, self).__init__()
        torch.manual_seed(42)

        self.num_layers = num_layers
        self.num_features = num_features
        self.embedding_size = embedding_size

        # Initial GCN layer
        self.initial_conv = GCNConv(num_features, embedding_size)

        # Additional GCN layers
        self.conv_layers = torch.nn.ModuleList([
            GCNConv(embedding_size,embedding_size) for _ in range(num_layers)
        ])

        # Output layer
        self.out = Linear(embedding_size, 1)

    def forward(self, x, edge_index, batch_index):
        """
        Perform a forward pass through the GCN model.

        Parameters:
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, num_features].
        
        edge_index : torch.Tensor
            Edge index matrix defining the graph structure, of shape [2, num_edges].
        
        batch_index : torch.Tensor
            Batch index vector to group nodes into batches, typically used for 
            mini-batch training.

        Returns:
        -------
        out : torch.Tensor
            The output predictions of shape [num_nodes, output_dim], where output_dim 
            is typically 1.

        hidden : torch.Tensor
            The final hidden node embeddings of shape [num_nodes, embedding_size], 
            capturing learned node representations.
        """
        # Initial Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.leaky_relu(hidden)

        # Hidden Conv layers
        for conv in self.conv_layers:
            hidden = conv(hidden, edge_index)
            hidden = F.leaky_relu(hidden)

        # Global Pooling (stack different aggregations)
        hidden = gmp(hidden, batch=batch_index)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden

