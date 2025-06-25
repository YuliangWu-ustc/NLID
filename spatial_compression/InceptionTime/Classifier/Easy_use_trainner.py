import os
from sklearn.metrics import accuracy_score
from os.path import dirname
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def eval_condition(iepoch,print_result_every_x_epoch):
    if (iepoch + 1) % print_result_every_x_epoch == 0:
        return True
    else:
        return False


def eval_model(model, dataloader):
    predict_list = np.array([])
    label_list = np.array([])
    for sample in dataloader:
        y_predict = model(sample[0])
        y_predict = y_predict.detach().cpu().numpy()
        y_predict = np.argmax(y_predict, axis=1)
        predict_list = np.concatenate((predict_list, y_predict), axis=0)
        label_list = np.concatenate((label_list, sample[1].detach().cpu().numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    return acc


def save_to_log(sentence, Result_log_folder, dataset_name):
    father_path = Result_log_folder + dataset_name
    if not os.path.exists(father_path):
        os.makedirs(father_path)
    path = father_path + '/' + dataset_name + '_.txt'
    print(path)
    with open(path, "a") as myfile:
        myfile.write(sentence + '\n')

class Easy_use_trainner():
    
    def __init__(self,
                 Result_log_folder, 
                 dataset_name, 
                 device, 
                 max_epoch = 2000, 
                 batch_size=16,
                 print_result_every_x_epoch = 50,
                 minium_batch_size = 2,
                 lr = None
                ):
        
        super(Easy_use_trainner, self).__init__()
        
        if not os.path.exists(Result_log_folder +dataset_name+'/'):
            os.makedirs(Result_log_folder +dataset_name+'/')
        Initial_model_path = Result_log_folder +dataset_name+'/'+dataset_name+'initial_model'
        model_save_path = Result_log_folder +dataset_name+'/'+dataset_name+'Best_model'
        

        self.Result_log_folder = Result_log_folder
        self.dataset_name = dataset_name        
        self.model_save_path = model_save_path
        self.Initial_model_path = Initial_model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.print_result_every_x_epoch = print_result_every_x_epoch
        self.minium_batch_size = minium_batch_size
        
        if lr == None:
            self.lr = 0.001
        else:
            self.lr = lr
        self.Model = None
    
    def get_model(self, model):
        self.Model = model.to(self.device)
        
        
    def fit(self, X_train, y_train, X_val, y_val):

        print('code is running on ',self.device)
        
        # covert numpy to pytorch tensor and put into gpu
        X_train = torch.from_numpy(X_train)
        X_train.requires_grad = False
        X_train = X_train.to(self.device)
        y_train = torch.from_numpy(y_train).to(self.device)
        
        
        X_test = torch.from_numpy(X_val)
        X_test.requires_grad = False
        X_test = X_test.to(self.device)
        y_test = torch.from_numpy(y_val).to(self.device)
        
        
        # add channel dimension to time series data
        if len(X_train.shape) == 2:
            X_train = X_train.unsqueeze_(1)
            X_test = X_test.unsqueeze_(1)
        
        # save_initial_weight
        torch.save(self.Model.state_dict(), self.Initial_model_path)
        
        
        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.Model.parameters(),lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)
        
        # build dataloader
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=max(int(min(X_train.shape[0] / 10, self.batch_size)),self.minium_batch_size), shuffle=True)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_train.shape[0] / 10, self.batch_size)),self.minium_batch_size), shuffle=False)
        
        
        self.Model.train()   
        
        for i in range(self.max_epoch):
            for sample in train_loader:
                optimizer.zero_grad()
                y_predict = self.Model(sample[0])
                loss = criterion(y_predict, sample[1])
                loss.backward()
                optimizer.step()
            scheduler.step(loss)
            
            if eval_condition(i,self.print_result_every_x_epoch):
                for param_group in optimizer.param_groups:
                    print('epoch =',i, 'lr = ', param_group['lr'])
                torch.set_grad_enabled(False)
                self.Model.eval()
                acc_train = eval_model(self.Model, train_loader)
                acc_test = eval_model(self.Model, test_loader)
                self.Model.train()
                torch.set_grad_enabled(True)
                print('train_acc=\t', acc_train, '\t test_acc=\t', acc_test, '\t loss=\t', loss.item())
                sentence = 'train_acc=\t'+str(acc_train)+ '\t test_acc=\t'+str(acc_test) 
                print('log saved at:')
                save_to_log(sentence,self.Result_log_folder, self.dataset_name)
                torch.save(self.Model.state_dict(), self.model_save_path)
         
        torch.save(self.Model.state_dict(), self.model_save_path)

        
        
    def predict(self, X_test):
        
        X_test = torch.from_numpy(X_test)
        X_test.requires_grad = False
        X_test = X_test.to(self.device)
        
        if len(X_test.shape) == 2:
            X_test = X_test.unsqueeze_(1)
        
        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_test.shape[0] / 10, self.batch_size)),2), shuffle=False)
        
        self.Model.eval()
        
        predict_list = np.array([])
        for sample in test_loader:
            y_predict = self.Model(sample[0])
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            
        return predict_list
    
    def load_model(self, model_path):
        self.Model.load_state_dict(torch.load(model_path))
        