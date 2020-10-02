import os
import h5py

import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import remove_self_loops

def load_feature(path):
    with h5py.File(path, 'r') as f:
        X = f['X'][()]
        eI = f['eI'][()]
        y = f['y'][()]
        eAttr = f['eAttr'][()]
    X = torch.tensor(X, dtype=torch.float32)
    eI = torch.tensor(eI, dtype=torch.long)
    # eI = remove_self_loops(eI)
    # eI = eI[0]
    y = torch.tensor(y, dtype=torch.long)
    eAttr = torch.tensor(eAttr, dtype=torch.float32)
    data = Data(x=X, edge_index=eI, y=y, edge_attr=eAttr)
    return data

class GRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GRDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return [x for x in self.root]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    
    def process(self):
        data_list = [load_feature(os.path.join(self.raw_dir, x)) for x in os.listdir(self.raw_dir)]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])