import gc

import os
import json
import random
import time
from shutil import copyfile

import numpy as np
import pandas as pd

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader

from data_utils import GRDataset
from model import Net
from utils import FocalLoss

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

with open('./params.json') as f:
    opt = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current device: ' + str(device))

save_dir = 'run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

os.mkdir(os.path.join(opt['opath'],save_dir))
output_path = os.path.join(opt['opath'],save_dir,'ckpts')
log_path = os.path.join(opt['opath'],save_dir,'logs')
config_path = os.path.join(opt['opath'],save_dir,'configs')
os.mkdir(output_path)
os.mkdir(log_path)
os.mkdir(config_path)
copyfile('./gr_train.py', os.path.join(config_path, 'gr_train.py'))
copyfile('./model.py', os.path.join(config_path, 'model.py'))
d = opt
print(json.dumps(d, indent=4))
with open(os.path.join(config_path, 'params.json'), 'w') as f:
    json.dump(d, f)

def main(opt):
    if opt['manual_seed'] is None:
        opt['manual_seed'] = random.randint(1, 10000)
        print('Random Seed: ', opt['manual_seed'])
        random.seed(opt['manual_seed'])
        torch.manual_seed(opt['manual_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opt['manual_seed'])
    
    if opt['class_weight'] is not None:
        loss_weight = torch.FloatTensor(opt['class_weight']).to(device)
    else:
        loss_weight = None
        
    if opt['gamma'] is not None:
        criterion = FocalLoss(alpha=loss_weight, gamma=opt['gamma'], reduction=True)
    else:
        criterion = CrossEntropyLoss(weight=loss_weight)
        
    files = []
    for file in os.listdir(opt['path']):
        files.append(file[:-3])
    
    trainings, validations = train_test_split(files, test_size=0.2)
    
    train_ids = [x for x in trainings]
    val_ids = [x for x in validations]
    
    train_dataset = GRDataset(opt['path'], train_ids)
    val_dataset = GRDataset(opt['path'], val_ids)
    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'], drop_last=True)
    
    tr_losses = np.zeros((opt['num_epochs'],))
    tr_accs = np.zeros((opt['num_epochs'],))
    val_losses = np.zeros((opt['num_epochs'],))
    val_accs = np.zeros((opt['num_epochs'],))
    
    model = Net(num_classes=opt['num_classes'], gnn_layers=opt['gnn_layers'],
                embed_dim=opt['embed_dim'], hidden_dim=opt['hidden_dim'],
                jk_layer=opt['jk_layer'], process_step=opt['process_step'], dropout=opt['dropout'])
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    best_val_loss = 1e6
        
    for epoch in range(opt['num_epochs']):
        s = time.time()
        
        model.train()
        losses = 0
        acc = 0
        
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print(data.y.squeeze())
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
            best_cnf = cnf
            torch.save(model.state_dict(), os.path.join(output_path, 'best_model.ckpt'))
        
        print('Epoch: {:03d} | time: {:.4f} seconds\n'
              'Train Loss: {:.4f} | Train accuracy {:.4f}\n'
              'Validation Loss: {:.4f} | Validation accuracy {:.4f} | Best {:.4f}'.format(epoch+1, time.time()-s, losses/(i+1),
                                acc/(i+1), v_losses/(j+1), v_acc/(j+1), best_val_loss))
        print('Validation confusion matrix:')
        print(cnf)
        
    np.save(os.path.join(log_path,'train_loss.npy', tr_losses))
    np.save(os.path.join(log_path,'train_acc.npy', tr_accs))
    np.save(os.path.join(log_path,'val_loss.npy', val_losses))
    np.save(os.path.join(log_path,'val_acc.npy', val_accs))
    np.save(os.path.join(log_path,'confusion_matrix.npy', best_cnf))
        
if __name__ == "__main__":
    main(opt)