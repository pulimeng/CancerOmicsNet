import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Linear, BatchNorm1d as BN
from aggin_conv import AGGINConv
from torch_geometric.nn import JumpingKnowledge, GlobalAttention, global_max_pool
from mys2s import Set2Set

class Net(torch.nn.Module):
    def __init__(self, num_classes, num_layers, feat_dim, embed_dim, jk_layer, process_step, dropout):
        super(Net, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(AGGINConv(Sequential(Linear(feat_dim, embed_dim),
                                                       ReLU(),
                                                       Linear(embed_dim, embed_dim),
                                                       ReLU(),
                                                       BN(embed_dim)), train_eps=True, dropout=self.dropout))
            else:
                self.convs.append(AGGINConv(Sequential(Linear(embed_dim, embed_dim),
                                                       ReLU(),
                                                       Linear(embed_dim, embed_dim),
                                                       ReLU(),
                                                       BN(embed_dim)), train_eps=True, dropout=self.dropout))

        if jk_layer.isdigit():
            jk_layer = int(jk_layer)
            self.jump = JumpingKnowledge(mode='lstm', channels=embed_dim, num_layers=jk_layer)
            self.gpl = (Set2Set(embed_dim, processing_steps=process_step))
            self.fc1 = Linear(2 * embed_dim, embed_dim)
            # self.fc1 = Linear(embed_dim, embed_dim)
            self.fc2 = Linear(embed_dim, num_classes)
        elif jk_layer == 'cat':
            self.jump = JumpingKnowledge(mode=jk_layer)
            self.gpl = (Set2Set(num_layers * embed_dim, processing_steps=process_step))
            self.fc1 = Linear(2 * embed_dim, embed_dim)
            # self.fc1 = Linear(num_layers * embed_dim, embed_dim)
            self.fc2 = Linear(embed_dim, num_classes)
        elif jk_layer == 'max':
            self.jump = JumpingKnowledge(mode=jk_layer)
            self.gpl = (Set2Set(embed_dim, processing_steps=process_step))
            self.fc1 = Linear(2 * embed_dim, embed_dim)
            # self.fc1 = Linear(embed_dim, embed_dim)
            self.fc2 = Linear(embed_dim, num_classes)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.gpl.reset_parameters()
        self.jump.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = self.gpl(x, batch)
        # x = global_max_pool(x, batch)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.fc2(x)

        return logits