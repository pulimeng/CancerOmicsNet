# GraphGR

GraphGR is a tool to predict drug (kinase inhibitors ONLY for now) response on a specific cell line given genetic information of the cell line, binding affinity of the drug to the proteins, and the disease association scores of the proteins. It utilizes heterogeneous data to construct a graph for each cell-line-drug combination (instance). The trained model can be used to predict the effect (currently a binary classification) of a drug on a given cell line.

If you find this tool useful, please cite our paper :)

Singha M, Pu L, Shawky M, Busch K, Wu H-C, Ramanujam J, Brylinski M (2020) GraphGR: A graph neural network to predict the effect of pharmacotherapy on the cancer cell growth. BioRxiv. https://www.biorxiv.org/content/10.1101/2020.05.20.107458v1

This README file is written by Limeng Pu.

# Dependencies

1. Non-deep-learning (data generation) related dependencies can be installed via `conda env create -f environment.yml`. Please change line 12 accordingly.
2. PyTorch Geometric latest version (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

The installation of Pytorch and Pytorch Geometric differs from system to system. Thus, for the installation instruction related to those libraries, please refer to the corresponding project site.

# Usage

The data generation and prediction/training are separated in this repo.

## Data Generation

The input data required for the model comes in the form of graphs (eventually converted to matrices), which consists of a list of nodes and a list of edges (edge features are not supported in the currently implementation), which can be downloaded from on the STRING databse (https://string-db.org/). The data generation steps include: graph generation with node feaures, graph reduction, and graph to matrix conversion. 

**Since the entire data generation process, especially the collection of node features (affinities, disease scores, and gene expressions), is quite cumbersome, thus the following procedure is ONLY for generating testing data (no labels!!). If you wish to generate your own training data (with class labels), you will need to collect GR values for them and follow the instruction @other repo.**

For the convenient of most users, we provide all the features we collected. The user only need to input the drug-cell-line combinations they wish to generate graphs for. To generate data for desired drug and cell-line, one first need to download and untar the already reduced graph from https://osf.io/enz69/. They are named, `reduced_node_tables` and `reduced_edge_tables`. Then run
<pre><code>python gr_datagen.py --i example_input --o output_folder</code></pre>
 - `--i` the file contains drug-cell-line pair you wish to generate data for, separated by commas. An example file in given as `example_input`. **Note that available drugs and cell-lines are also provided as drugs.lst and celllines.lst respectively.**
 - `--o` the folder you wish to store the generated data (.h5 files contain the matrices).

## Prediction

Once one finishes generating the desirable data, the prediction module can be carried out by running 
<pre><code>python gr_pred.py --m ./trained_model/graphgr_weights.ckpt --c ./trained_mode/configs.json --i your_data_folder --o your_output_file.</code></pre>
  - `--m` trained model weight file location.
  - `--c` trained model configuration file location.
  - `--i` input data folder location.
  - `--o` output file location.
  
A output will be produced in the form of:

| instance | drug | cellline | score | class |
|----------|------|----------|-------|-------|
|pazopanib_1321N1|pazopanib|1321N1|0.75794625|1|
|motesanib_1321N1|motesanib|1321N1|0.5741133|1|
|lestaurtinib_1321N1|lestaurtinib|1321N1|0.7808407|1|

where the score will represent the effectiveness of such drug on the provided cell line.

## Training

Before you train your own model, create two folders (`raw` and `processed`) under your data folder. Then move the data to `./raw`. To train the model using your own data, run 
<pre><code>python gr_train.py</code></pre>
It reads the configurations from the `params.json`.
 - `path` path to the training data folder.
 - `opath` path to the output folder.
 - `embed_dim` embedding dimension for the embedding of gene expressions and node types. **Not the embedding dimension of GNN.**
 - `hidden_dim` GNN hidden dimension.
 - `jk_layer` JumpingKnowledge layer parameter, can be 'cat', 'max', or any numbers.
 - `process_step` number of processing step for Set2Set readout layer.
 - `gamma` gamma for focal loss

All the results will be saved to the output folder, including loss, accuracies and best model weights.

# Dataset

The dataset we used for training and other auxilary files for data generation (`reduced_node_tables` and `reduced_edge_tables`) can be downloaded at https://osf.io/enz69/.
