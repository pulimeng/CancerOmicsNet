import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Linear, Embedding
from aggin_conv import AGGINConv
from torch_geometric.nn import JumpingKnowledge, Set2Set, BatchNorm as BN

class Net(torch.nn.Module):
    def __init__(self, num_classes, gnn_layers, embed_dim,
                 hidden_dim, jk_layer, process_step, dropout):
        super(Net, self).__init__()
        
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.embedding = Embedding(6, embed_dim)
        
        for i in range(gnn_layers):
            if i == 0:
                self.convs.append(AGGINConv(Sequential(Linear(2*embed_dim+2, hidden_dim),
                                                       ReLU(),
                                                       Linear(hidden_dim, hidden_dim),
                                                       ReLU(),
                                                       BN(hidden_dim)), train_eps=True))
            else:
                self.convs.append(AGGINConv(Sequential(Linear(hidden_dim, hidden_dim),
                                                       ReLU(),
                                                       Linear(hidden_dim, hidden_dim),
                                                       ReLU(),
                                                       BN(hidden_dim)), train_eps=True))
            
        if jk_layer.isdigit():
            jk_layer = int(jk_layer)
            self.jk = JumpingKnowledge(mode='lstm', channels=hidden_dim, gnn_layers=jk_layer)
            self.s2s = (Set2Set(hidden_dim, processing_steps=process_step))
            self.fc1 = Linear(2 * hidden_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, int(hidden_dim/2))
            self.fc3 = Linear(int(hidden_dim/2), num_classes)
        elif jk_layer == 'cat':
            self.jk = JumpingKnowledge(mode=jk_layer)
            self.s2s = (Set2Set(gnn_layers * hidden_dim, processing_steps=process_step))
            self.fc1 = Linear(2 * gnn_layers * hidden_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, int(hidden_dim/2))
            self.fc3 = Linear(int(hidden_dim/2), num_classes)
        elif jk_layer == 'max':
            self.jk = JumpingKnowledge(mode=jk_layer)
            self.s2s = (Set2Set(hidden_dim, processing_steps=process_step))
            self.fc1 = Linear(2 * hidden_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, int(hidden_dim/2))
            self.fc3 = Linear(int(hidden_dim/2), num_classes)
        
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jk.reset_parameters()
        self.s2s.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Embedding the categorical values from Gene expression and Node type
        xc = x[:,:2].type(torch.long)
        ems = self.embedding(xc)
        ems = ems.view(-1, ems.shape[1]*ems.shape[2])
        x = torch.cat((ems, x[:,2:]), dim=1)
        
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            xs += [x]
        x = self.jk(xs)
        x = self.s2s(x, batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        return logits