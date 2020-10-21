import argparse

import os
import time
import h5py
import json

import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import importlib.util

import random

seed = 12345
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
        
def load_feature(path):
    with h5py.File(path, 'r') as f:
        X = f['X'][()]
        eI = f['eI'][()]
        y = f['y'][()]
    X = torch.tensor(X, dtype=torch.float32)
    X[:,0] = X[:,0]+1 # add 1 to Gene exp to make range 0-2 for embedding purposes
    X[:,1] = X[:,1]+3 # add 3 to Node type to make range 3-5 for embedding purposes
    eI = torch.tensor(eI, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    data = Data(x=X, edge_index=eI, y=y, edge_attr=None)
    return data

def main(opt):
    weight_file = opt.m
    config_file = opt.c
    data_folder = opt.i
    output_file = opt.o
    
    spec = importlib.util.spec_from_file_location('Net', './model.py')
    mo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mo)
    
    with open(config_file) as f:
        opt = json.load(f)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: ' + str(device))
    
    model = mo.Net(num_classes=opt['num_classes'], gnn_layers=opt['gnn_layers'],
                   embed_dim=opt['embed_dim'], hidden_dim=opt['hidden_dim'],
                   jk_layer=opt['jk_layer'], process_step=opt['process_step'], dropout=opt['dropout'])
    
    model.to(device)
    print(model)
    
    output_df = pd.DataFrame(columns=['instance','drug','cellline','score','class'])
    
    cnt = 0
    print('Loading model...')
    model.load_state_dict(torch.load(weight_file))
    model.eval()
    print('Running prediction...')
    for file in os.listdir(data_folder):
        ins = file[:-3]
        output_df.at[cnt, 'instance'] = ins
        output_df.at[cnt, 'drug'] = ins.split('_')[0]
        output_df.at[cnt, 'cellline'] = ins.split('_')[1]
        data = load_feature(os.path.join(data_folder, file))
        data.batch = torch.zeros((data.x.shape[0],), dtype=torch.long)
        data = data.to(device)
        output = model(data)
        proba = F.softmax(output, dim=1)
        pred = output.data.cpu().numpy().argmax(axis=1)
        output_df.at[cnt, 'score'] = proba.data.cpu().numpy()[:,1][0]
        output_df.at[cnt, 'class'] = pred[0]
        cnt += 1
        del data
        del output
    
    print('Saving results...')
    output_df.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=str, help='Trained model weights')
    parser.add_argument('--c', type=str, help='Trained model configurations')
    parser.add_argument('--i', type=str, help='Data folder')
    parser.add_argument('--o', type=str, help='Output filename')
    opt = parser.parse_args()
    main(opt)
