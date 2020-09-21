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

