# Data Preprocessing for GraphGR

The details of the data preprocessing for GraphGR is presented as follows.

1. Graph generation

    The network is represented as a node table and an edge table. The node table takes the following format:
    
    |   node_id   |   feature_1   |   feature_2   |   ...   |   feature_m   |
    |:---:|:---:|:---:|:---:|:---:|
    | ENSEMBLID_1 | x1_1 | x1_2 | ... | x1_m |
    | ENSEMBLID_2 | x2_1 | x2_2 | ... | x2_m |
    | ... | ... | ... | ... | ... | ... |
    | ENSEMBLID_n | xn_1 | xn_2 | ... | xn_m |

    Note that we use the ensemble id as the node id for proteins. The node table is the table provided by the user. It contains any features you wish to use for your purpose. However, the node presented in the node table must also be presented in the edge table, which is provided in this repo. Otherwise, the network will have an isolated node, where no information propagation occurs regarding such nodes. This makes such nodes useless in the learning process since there is no information exchange with any other nodes.

    The sample edge table is shown as follow:
    
    |   node1_id   |   node2_id   |   feature_1   |
    |:---:|:---:|:---:|
    | ENSEMBLID_1 | ENSEMBLID_2 | conf_1_12 |
    | ENSEMBLID_1 | ENSEMBLID_3 | conf_1_13 |
    | ... | ... | ... |
    | ENSEMBLID_n | ENSEMBLID_k | conf_1_nk |

    The edge table is the connections provided by the STRING database. It often require some preprocess (connected component) based on particular applications.
    
    These two tables are often enough to represent the PPI network.
    
2. Graph reduction

    The network reduction heavily relies on the applications/purpose of the study since it is related to the nature of the network. In this example, we presented a simple case. We reduce the network by contracting edges (merging nodes), whose two nodes (proteins) belong to the <strong>same pathway</strong> AND have the <strong>same gene expression</strong> AND <strong>neither of them are kinases</strong>. In order to do that, we loaded the following data:
    
    - `./data/master_edge_table.csv` -- The preprocessed yet UNREDUCED edge table from STRING as described above. NOTE that this is not the original string. Only interactions with score above 500 is retained and some isolated proteins are removed.
    - `./data/master_node_table.csv` -- The original (unreduced) node table with the format as aforementioned. Contains many features.
    - `./data/pathway_info.csv` -- Pathway information of all the proteins in the processed STIRING. "unknown" is used for the missing values.
    - `./data/kinases.pkl` -- A list of all kinases.
    
    The reduction procedure performed in this example can be summarized as following steps.
    
    - Find UNCONTRACTABLE edges.
    - Remove UNCONTRACABLE edges on a copy of the original network.
    - Find all connected components C = {c_1, c_2, ... , c_l} with more than two nodes in G'. This can be done easily using networkx.
    - Replace all nodes in each connected component with one virtual node v_i, where i = 1,2,3,...,l. Note that some of the node features should be replaced with median/max/mean of the original nodes associated with corresponding virtual node. This should be done in both node table and edge table.
    - Remove redundant nodes, edges in the corresponding tables.
    - Generate new graph from the reduced edge table. 
    
    A flowchart of the procedure is presented in the figure below.
    
    ![eg_image](https://github.com/pulimeng/GraphGR/tree/master/preprocessing/image/reduction_flow.png)
    
    The script will produce four outputs:
    
    - `reduced_network.gexf` -- A graph representation of the network. Contains all the features (both nodes' and edges').
    - `reduced_node_table.csv` -- One extra feature will be produced to represent the reduction procedure. The kinase nodes will have a value of 2 while the virtual nodes will have a value of 0. And the non-kinase unreduced nodes will have a value of 1.
    - `reduced_edge_table.csv` -- Same format as the original edge table.
    - `reduction_records.csv` -- A record to keep track of every node's final destination in the reduction procedure.
