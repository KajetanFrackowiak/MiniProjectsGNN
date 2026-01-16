### GAT GNN Implementation

The project tries to reimplement the concepts presented in the original GAT paper: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) using PyTorch Geometric library.

### Datasets

The project uses standard transductive 
- [Cora](https://linqs.soe.ucsc.edu/data)
- [CiteSeer](https://linqs.soe.ucsc.edu/data)
- [PubMed](https://linqs.soe.ucsc.edu/data)

and the inductive node classification datasets from PyTorch Geometric used in the original paper:
- [PPI](https://arxiv.org/abs/1707.04638) dataset used in the original paper for node classification tasks.

### Results
Cora dataset:
- Test Accuracy: 0.8

<img src="training_plots/training_validation_loss_Cora_20260116_192028.png" width="600" alt="Cora training-validation loss plot">

CiteSeer dataset:
- Test Accuracy: 0.67

<img src="training_plots/training_validation_loss_CiteSeer_20260116_192042.png" width="600" alt="CiteSeer training-validation loss plot">

PubMed dataset:
- Test Accuracy: 0.77

<img src="training_plots/training_validation_loss_PubMed_20260116_192103.png" width="600" alt="PubMed training-validation loss plot">

PPI dataset:
- Test F1 Score: ~0.395
<img src="training_validation_loss_PPI_20260116_192618.png" width="600" alt="PPI training-validation loss plot">

