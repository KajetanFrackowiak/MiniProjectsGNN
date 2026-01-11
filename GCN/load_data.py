import torch 
from torch_geometric.datasets import Planetoid

def load_data(dataset_name='Cora', root='./data'):
    """
    Load a graph dataset using PyTorch Geometric.

    Parameters:
    - dataset_name (str): Name of the dataset to load. Default is 'Cora'.
    - root (str): Root directory where the dataset should be saved. Default is './data'.

    Returns:
    - data (torch_geometric.data.Data): The loaded graph data object.
    """
    # dataset.shape = (num_nodes, num_node_features)
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]  # Get the first graph object from the dataset
    return data
