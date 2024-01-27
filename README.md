# MARIO: Model Agnostic Recipe for Improving OOD Generalization of Graph Contrastive Learning

**Official implementation of paper**  <br>[MARIO: Model Agnostic Recipe for Improving OOD Generalization of Graph Contrastive Learning](https://arxiv.org/abs/2307.13055) <br>

Yun Zhu, Haizhou Shi, Zhenshuo Zhang, Siliang Tang†

In WWW 2024

## Setup

```
conda create -n MARIO python==3.9
conda activate MARIO 
conda install pytorch==2.1.0 torchaudio==2.1.0 cudatoolkit=12.1 -c pytorch -c conda-forge
pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install munch ruamel_yaml cilog gdown dive_into_graphs tensorboard rich
```