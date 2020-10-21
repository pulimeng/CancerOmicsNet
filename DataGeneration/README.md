# Data Generation for GraphGR

The details of the data generation for GraphGR is presented as follows. **Note the following procedure is ONLY for generating testing data. If you wish to generate your own training data, you will need to collect GR values for them, which is not provided here, and follow the same method described below.**
    

# Prerequisites

1. Python 3.7.*
2. Pandas 1.0 or higher
3. Networkx 2.4 or higher

# Usage

The entire data generation process, especially the collection of node features (affinities, disease scores, and gene expressions), is quite cumbersome. Thus, for the convenient of the users, we provide all the features we collected. The user only need to input the drug-cellline combinations they wish to generate graphs for.
