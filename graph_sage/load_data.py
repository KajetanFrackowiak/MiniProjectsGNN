from torch_geometric.datasets import PPI

def load_data(root='./data'):
    """
    Returns train_loader, val_loader, test_loader
    """
    dataset_train = PPI(root=root, split='train')
    dataset_val = PPI(root=root, split='val')
    dataset_test = PPI(root=root, split='test')

    return dataset_train, dataset_val, dataset_test
