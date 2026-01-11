### Graph-Convolution Network (GCN) for Node Classification

This repository contains an implementation of a Graph-Convolution Network (GCN) for node classification tasks using PyTorch and PyTorch Geometric. The project tries to reimplement 
the original GCN model proposed by Kipf and Welling in their paper [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

### Datasets

The implementation supports popular graph datasets such as Cora, Citeseer, and Pubmed, which can be easily loaded using PyTorch Geometric's dataset utilities.

### Results

Cora
- Accuracy: 81.5%

![Cora training-validation plot](training_plots/training_validation_loss_Cora_20260111_195530.png)

Citeseer
- Accuracy: 70.3%

![Citeseer training-validation plot](training_plots/training_validation_loss_Citeseer_20260111_195559.png)

Pubmed
- Accuracy: 79.0%

![Pubmed training-validation plot](training_plots/training_validation_loss_Pubmed_20260111_195633.png)