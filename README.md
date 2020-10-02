# GraphGR

GraphGR is a tool to predict drug response on a specific cell line given genetic information of the cell line, binding affinity of the drug to the proteins, and the disease association scores of the proteins. It utilizes and combines heterogeneous data to construct a graph for each cell-line-drug combination (instance)

If you find this tool useful, please cite our paper :)

Singha M, Pu L, Shawky M, Busch K, Wu H-C, Ramanujam J, Brylinski M (2020) GraphGR: A graph neural network to predict the effect of pharmacotherapy on the cancer cell growth. BioRxiv. https://www.biorxiv.org/content/10.1101/2020.05.20.107458v1

This README file is written by Limeng Pu.

# Prerequisites

1. Python 3.7.*
2. Pytorch 1.4.0 or higher
3. PyTorch Geometric latest version (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
4. Pandas 1.0 or higher
5. Networkx 2.4 or higher
6. CUDA if you'd like to run on GPU(s)

For the installation instruction please refer to the corresponding project site.

# Usage

This repo provides data processing, prediction, and training modules if you'd like to train on your own dataset. 

# Data Preprocessing

The input data required for the model comes in the form of graphs, which consists of a list of nodes and a list of edges (edge features are not supported in the currently implementation), which is provided in this repo based on the STRING databse (https://string-db.org/). The processing steps include: graph generation with node feaures, graph reduction, and graph to matrix conversion. 

1. Graph generation: 
input --> node table

    |   node_id   |   feature_1   |   feature_2   |   ...   |   feature_m   |
    |:---:|:---:|:---:|:---:|:---:|
    | ENSEMBLID_1 | x1_1 | x1_2 | ... | x1_m |
    | ENSEMBLID_2 | x2_1 | x2_2 | ... | x2_m |
    | ... | ... | ... | ... | ... | ... |
    | ENSEMBLID_n | xn_1 | xn_2 | ... | xn_m |
    
Note that we use the ensemble id as the node id.
output --> Networkx.Graph
