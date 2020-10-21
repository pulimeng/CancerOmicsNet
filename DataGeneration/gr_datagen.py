import argparse

import os
from time import time
import h5py

import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sps

pd.options.mode.chained_assignment = None  # default='warn'

def main(opt):
    with open(opt.i) as f:
        inputs = f.read().strip('\n').replace(' ', '')
            
    combinations = inputs.split(',')
    total_affinity = pd.read_csv('./affinities.csv')
    
    try:
        os.mkdir(opt.o)
        print('Creating output folder')
    except:
        print('Output folder exists')
    
    ds = []
    with open('drugs.lst') as f:
        for item in f.readlines():
            ds.append(item.strip('\n'))
    
    cs = []
    with open('celllines.lst') as f:
        for item in f.readlines():
            cs.append(item.strip('\n'))
    
    st = time()
    cnt = 0
    for i, comb in enumerate(combinations):
        drug = comb.split('_')[0]
        cl = comb.split('_')[1]
        if drug not in ds:
            print('Drug not found for {}'.format(comb))
            cnt += 1
            continue
        if cl not in cs:
            print('Cell line not found for {}'.format(comb))
            cnt += 1
            continue
        affinity = total_affinity[total_affinity['Drug_Name'] == drug]
        affinity.sort_values(by='Affinity', inplace=True, ascending=False)
        affinity.drop_duplicates('ensembl', inplace=True)
        affinity = affinity.set_index('ensembl')
    
        node_table = pd.read_csv(os.path.join('./reduced_node_tables', cl+'.nodes'))
        node_table = node_table.set_index('ensembl')
        node_table.update(affinity)
        node_table = node_table.reset_index()
        
        edge_table = pd.read_csv(os.path.join('./reduced_edge_tables', cl+'.edges.gz'))
        G = nx.from_pandas_edgelist(edge_table, source='protein1', target='protein2',
                                        edge_attr = edge_table.columns.tolist()[2])
        """
        The following is important
        Make sure the order of nodes in node table matches the node order in the nx.Graph
        Otherwise, the feature matrix and the adjacency matrix will have mismatched entries
        """
        node_ids = node_table['ensembl'].tolist()
        G_node_ids = list(G.nodes)
        new_index = [node_ids.index(x) for x in G_node_ids]
        node_table = node_table.reindex(index=new_index)
        order = node_table['ensembl'].tolist()
        if order != G_node_ids:
            print('Error at {}_{}'.format(drug, cl))
        
        feature = node_table.loc[:,['gene_exp',  'kinaseness', 'Affinity', 'disease_score']]
        feature_matrix = feature.to_numpy()
        A = nx.to_numpy_array(G)
        coo_A = np.array([sps.coo_matrix(A).row, sps.coo_matrix(A).col])
        edge_index = coo_A.astype(int)
        label = -1 # placeholder for labels
        print(comb)
        with h5py.File(os.path.join(opt.o, '{}.h5'.format(comb)), 'w') as f:
            f.create_dataset('X', data=feature_matrix, compression='gzip', compression_opts=9)
            f.create_dataset('A', data=A, compression='gzip', compression_opts=9)
            f.create_dataset('eI', data=edge_index, compression='gzip', compression_opts=9)
            f.create_dataset('y', data=np.array(label,dtype=int).reshape((1,1)), compression='gzip', compression_opts=9)
                
    print('Total data generated {} in {:.4f} seconds.'.format(i+1-cnt, time() - st))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, help='Input file')
    parser.add_argument('--o', type=str, help='Output folder')
    opt = parser.parse_args()
    main(opt)

