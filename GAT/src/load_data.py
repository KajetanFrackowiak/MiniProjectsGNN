from typing import cast

from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import PPI, Planetoid


def load_data(
    dataset_name: str = "PPI", root: str = "./data"
) -> tuple[Data | Dataset, Data | Dataset, Data | Dataset]:
    """
    Returns train_data, val_data, test_data
    """
    if dataset_name == "Cora" or dataset_name == "CiteSeer" or dataset_name == "PubMed":
        dataset = Planetoid(root=root, name=dataset_name)
        data = cast(Data, dataset[0])

        return data, data, data

    elif dataset_name == "PPI":
        dataset_train = PPI(root=root, split="train")
        dataset_val = PPI(root=root, split="val")
        dataset_test = PPI(root=root, split="test")

        return dataset_train, dataset_val, dataset_test

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Supported datasets: 'Cora', 'CiteSeer', 'PubMed', 'PPI'"
        )
