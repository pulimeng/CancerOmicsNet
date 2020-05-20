import gc

import argparse
import os
import json
import random
import time
from shutil import copyfile

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import DataLoader

from mydataset import MyDataset
from model import Net

from sklearn.metrics import accuracy_score, confusion_matrix

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,
                    help='path to the feature data')
parser.add_argument('--lpath', type=str,
                    help='path to the file that holds all the instance id and its corresponding labels. used for cross validation')
parser.add_argument('--opath', type=str, default='./results/',
                    help='output path')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size')
parser.add_argument('--num_layers', type=int, default=3,
                    help='number of conv layers')
parser.add_argument('--embed_dim', type=int,
                    help='embedding dimension')
parser.add_argument('--feat_dim', type=int,
                    help='number of node features')
parser.add_argument('--jk_layer', type=str,
                    help='jump knowledge mode, if integer convert to lstm layers')
parser.add_argument('--process_step', type=int,
                    help='set2set process steps')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='l-2 regularization weight decay')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='probability of dropout')
parser.add_argument('--num_classes', type=int, default=2,
                    help='number of classes')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--manual_seed', type=int,
                    help='manual seed')

opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current device: ' + str(device))

header = opt.path.split('/')[-3]
cv_info = opt.lpath.split('/')[-1].split('_')[1][:-4]
save_dir = '{}_{}_lr{}_epoch{}_bs{}_layer{}_embed{}_jk{}_s2s{}_dropout{}_ltwo{}'.format(header, cv_info,
                                                                 opt.lr, opt.num_epochs, opt.batch_size,
                                                                 opt.num_layers, opt.embed_dim, opt.jk_layer,
                                                                 opt.process_step, opt.dropout, opt.weight_decay
                                                                 )
os.mkdir(os.path.join(opt.opath,save_dir))
output_path = os.path.join(opt.opath,save_dir,'ckpts')
log_path = os.path.join(opt.opath,save_dir,'logs')
config_path = os.path.join(opt.opath,save_dir,'configs')
os.mkdir(output_path)
os.mkdir(log_path)
os.mkdir(config_path)
copyfile('./aggin_conv.py', os.path.join(config_path, 'aggin_conv.py'))
copyfile('./main.py', os.path.join(config_path, 'main.py'))
copyfile('./model.py', os.path.join(config_path, 'model.py'))
d = vars(opt)
print(json.dumps(d, indent=4))
with open(os.path.join(config_path, 'configurations.json'), 'w') as f:
    json.dump(d, f)

def main(opt):
    if opt.manual_seed is None:
        opt.manual_seed = random.randint(1, 10000)
        print('Random Seed: ', opt.manual_seed)
        random.seed(opt.manual_seed)
        torch.manual_seed(opt.manual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opt.manual_seed)
            
    model = Net(num_classes=opt.num_classes, num_layers=opt.num_layers, feat_dim=opt.feat_dim, embed_dim=opt.embed_dim,
                jk_layer=opt.jk_layer, process_step=opt.process_step, dropout=opt.dropout)
    print(model)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
        
    df = pd.read_csv(opt.lpath)
    nfold = len(df['group'].unique())
    
    dataset = MyDataset(opt.path)
    
    for k in range(nfold):
        trainings = df[df['group']!=k+1]['instance'].tolist()
        validations = df[df['group']==k+1]['instance'].tolist()
        total = [f[:-3] for f in os.listdir(os.path.join(opt.path, 'raw'))]
        train_ids = [total.index(x) for x in trainings]
        val_ids = [total.index(x) for x in validations]
        
        train_ids = torch.tensor(train_ids, dtype=torch.long)
        val_ids = torch.tensor(val_ids, dtype=torch.long)
        train_dataset = dataset[train_ids]
        val_dataset = dataset[val_ids]
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True)
        
        tr_losses = np.zeros((opt.num_epochs,))
        tr_accs = np.zeros((opt.num_epochs,))
        val_losses = np.zeros((opt.num_epochs,))
        val_accs = np.zeros((opt.num_epochs,))
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay, nesterov=True)
        best_val_loss = 1e6
        
        print('===================Fold {} starts==================='.format(k+1))
        for epoch in range(opt.num_epochs):
            s = time.time()
            
            model.train()
            losses = 0
            acc = 0
            
            for i, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data.y.squeeze())
                loss.backward()
                optimizer.step()
                
                y_true = data.y.squeeze().cpu().numpy()
                y_pred = output.data.cpu().numpy().argmax(axis=1)
                acc += accuracy_score(y_true, y_pred)*100
                losses += loss.data.cpu().numpy()

            tr_losses[epoch] = losses/(i+1)
            tr_accs[epoch] = acc/(i+1)
            
            model.eval()
            v_losses = 0
            v_acc = 0
            y_preds = []
            y_trues = []
            
            for j, data in enumerate(val_loader):
                data = data.to(device)
                with torch.no_grad():
                    output = model(data)
                    loss = criterion(output, data.y.squeeze())
                
                y_pred = output.data.cpu().numpy().argmax(axis=1)
                y_true = data.y.squeeze().cpu().numpy()
                y_trues += y_true.tolist()
                y_preds += y_pred.tolist()
                v_acc += accuracy_score(y_true, y_pred)*100
                v_losses += loss.data.cpu().numpy()
                
            cnf = confusion_matrix(y_trues, y_preds)        
            val_losses[epoch] = v_losses/(j+1)
            val_accs[epoch] = v_acc/(j+1)
            
            current_val_loss = v_losses/(j+1)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(model.state_dict(), os.path.join(output_path, 'best_model_fold{}.ckpt'.format(k+1)))
            
            print('Epoch: {:03d} | time: {:.4f} seconds\n'
                  'Train Loss: {:.4f} | Train accuracy {:.4f}\n'
                  'Validation Loss: {:.4f} | Validation accuracy {:.4f} | Best {:.4f}'.format(epoch+1, time.time()-s, losses/(i+1),
                                    acc/(i+1), v_losses/(j+1), v_acc/(j+1), best_val_loss))
            print('Validation confusion matrix:')
            print(cnf)
            
        print('===================Fold {} ends==================='.format(k+1))
        np.save(os.path.join(log_path,'train_loss_{}.npy'.format(k+1)), tr_losses)
        np.save(os.path.join(log_path,'train_acc_{}.npy'.format(k+1)), tr_accs)
        np.save(os.path.join(log_path,'val_loss_{}.npy'.format(k+1)), val_losses)
        np.save(os.path.join(log_path,'val_acc_{}.npy'.format(k+1)), val_accs)
        
#        break
        
if __name__ == "__main__":
    main(opt)