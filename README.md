# Make Graph Neural Networks Great Again: A Generic Integration Paradigm of Topology-Free Patterns for Traffic Speed Prediction (IJCAI 2024)
This is the origin Pytorch implementation of [Make Graph Neural Networks Great Again: A Generic Integration Paradigm of Topology-Free Patterns for Traffic Speed Prediction]() 

## Key Points of DCST
### 1.Spatial-Temporal Scale Generation
**Figure 1.** Spatial-Temporal Scale. **(a) Spatial Scale:** The features of nodes located in the same grid are aggregated, and different scales are divided according to the size of the grid. **(b) Temporal Scale:** For each node, aggregate the features of time points within the same temporal segment and divide them into different scales based on the length of the temporal segment. 

### 2.Dual Cross-Scale Transformer
**Figure 2.** Dual Cross-Scale Transformer is composed of an Embedding Layer, a Temporal Transformer, a Spatial Transformer, and a Prediction Layer.

### 3.Integration of Topology-regularized and Cross-Scale Topology-free Patterns
**Figure 3.** An illustration of the integration process of topology regularized/-free patterns. The integration process follows the teacher-student paradigm, where the GNN-based model is taken as the **teacher model (topology-regularized patterns)**, and the Dual Cross-Scale Transformer is taken as the **student model (topologyfree patterns)**. During the process, the GNN-based model has been pre-trained and kept fixed. The integration is conducted by jointly optimizing "Soft Loss" and "Hard Loss".

## Requirements
Our code is based on Python version 3.8.19 and PyTorch version 1.10.0. Please make sure you have installed Python and PyTorch correctly. Then you can install all the dependencies with the following command by pip:
`pip install -r requirements.txt`


