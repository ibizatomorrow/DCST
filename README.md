# Make Graph Neural Networks Great Again: A Generic Integration Paradigm of Topology-Free Patterns for Traffic Speed Prediction (IJCAI 2024)
This is the origin Pytorch implementation of [Make Graph Neural Networks Great Again: A Generic Integration Paradigm of Topology-Free Patterns for Traffic Speed Prediction]() 

## Key Points of DCST
### 1.Spatial-Temporal Scale Generation
**Figure 1.** Spatial-Temporal Scale. **(a) Spatial Scale:** The features of nodes located in the same grid are aggregated, and different scales are divided according to the size of the grid. **(b) Temporal Scale:** For each node, aggregate the features of time points within the same temporal segment and divide them into different scales based on the length of the temporal segment. 

### 2.Dual Cross-Scale Transformer
**Figure 2.** Dual Cross-Scale Transformer is composed of an Embedding Layer, a Temporal Transformer, a Spatial Transformer, and a Prediction Layer.

### 3.Integration of Topology-regularized and Cross-Scale Topology-free Patterns
**Figure 3.** An illustration of the integration process of topology regularized/-free patterns. The integration process follows the teacher-student paradigm, where the GNN-based model is taken as the **teacher model (topology-regularized patterns)**, and the Dual Cross-Scale Transformer is taken as the **student model (topology-free patterns)**. During the process, the GNN-based model has been pre-trained and kept fixed. The integration is conducted by jointly optimizing "Soft Loss" and "Hard Loss".

## Requirements
Our code is based on Python version 3.8.19 and PyTorch version 1.10.0. Please make sure you have installed Python and PyTorch correctly. Then you can install all the dependencies with the following command by pip: 
```python
pip install -r requirements.txt
```

## Reproducibility
1. The dataset has already been stored in the data folder. The dataset directory will look like this:
<pre> ```Data ├── METRLA ├── PEMSBAY ├── PEMSD7 ``` </pre>


2. To get results of DCST on METRLA dataset with GWNet as Teacher model, run:
```python
cd model/
python train.py --T_model GWNet --dataset METRLA --gpu 0 --kd_weight 0.5
```
The model will be automatically trained and tested. The trained model will be saved in folder `saved_models/` and Training logs will be saved in folder `logs/`.

We describe parameters in detail:
| Parameter name | Description of parameter |
|----------------|--------------------------|
| `T_model`         | Teacher model which capture topology-regularized patterns, choice: `[GWNet, MTGNN, AGCRN, STGCN, DCRNN]`|
| `dataset`         | The dataset name, choice: `[MTRELA, PEMSBAY, PEMSD7]` |
| `kd_weight`         | Percentage of topology-free patterns, can be ratio (e.g. 0.3, 0.5, 0.7) |

## Citation
If you find this repository useful in your research, please cite:
```python
@inproceedings{zhou2024make,
  title={Make graph neural networks great again: a generic integration paradigm of topology-free patterns for traffic speed prediction},
  author={Zhou, Yicheng and Wang, Pengfei and Dong, Hao and Zhang, Denghui and Yang, Dingqi and Fu, Yanjie and Wang, Pengyang},
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  pages={2607--2615},
  year={2024}
}
```







