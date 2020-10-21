# GraphGR

GraphGR is a tool to predict drug (kinase inhibitors ONLY for now) response on a specific cell line given genetic information of the cell line, binding affinity of the drug to the proteins, and the disease association scores of the proteins. It utilizes heterogeneous data to construct a graph for each cell-line-drug combination (instance). The trained model can be used to predict the effect (currently a binary classification) of a drug on a given cell line.

If you find this tool useful, please cite our paper :)

Singha M, Pu L, Shawky M, Busch K, Wu H-C, Ramanujam J, Brylinski M (2020) GraphGR: A graph neural network to predict the effect of pharmacotherapy on the cancer cell growth. BioRxiv. https://www.biorxiv.org/content/10.1101/2020.05.20.107458v1

This README file is written by Limeng Pu.

# Dependencies

1. Non-deep-learning (data generation) related dependencies can be installed via `conda env create -f environment.yml`. Please change line 12 accordingly.
2. Pytorch 1.6.0
3. PyTorch Geometric latest version (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
4. CUDA if you'd like to run on GPU(s)

The installation of Pytorch and Pytorch Geometric differs from system to system. Thus, for the installation instruction related to those libraries, please refer to the corresponding project site.

# Usage

The data generation and prediction/training are separated in this repo.

# Data Generation

The input data required for the model comes in the form of graphs, which consists of a list of nodes and a list of edges (edge features are not supported in the currently implementation), which is provided in this repo based on the STRING databse (https://string-db.org/). The generation steps include: graph generation with node feaures, graph reduction, and graph to matrix conversion. **Since the entire data generation process, especially the collection of node features (affinities, disease scores, and gene expressions), is quite cumbersome. Thus, for the convenient of the users, we provide all the features we collected. The user only need to input the drug-cellline combinations they wish to generate graphs for. Note the following procedure is ONLY for generating testing data. If you wish to generate your own training data, you will need to collect GR values for them, which is not provided here, and follow the same method described below.**



The detailed information can be found in `./preprocessing`.

1. Graph generation: 

    Generate a graph representation of the input data based on the node table provided by the user and the edge table provided in the repo. The resulting graph representation will be used for the graph reduction in the next step.

    input --> node table (.csv) Note that we use the ensemble id as the node id.
    
    output --> graph (.gexf, can be read in Networkx).

2. Graph reduction:

    Reduce the original graph representation to a more feature rich form for better learning performance.

    input --> node table (.csv), graph (.gexf), additional reduction rules.

    output --> reduced node table (.csv), reduced edge table (.csv), reduced graph (.gexf), reduction records (.csv).

3. Graph to matrix conversion

    Convert the graph to matrix reprsentation as the final input to the model.

    input --> reduced node table (.csv), reduced graph (.gexf).

    output --> matrices (.h5) including node features matrix (N x d), adjacency matrix (N x N), edge indices (2 x E).

# Prediction

Once one finished generating the input data, the prediction can be produced by 
