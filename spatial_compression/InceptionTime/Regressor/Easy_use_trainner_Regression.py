import os
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# 修改评估函数为回归指标
def eval_model(model, dataloader):
    predict_list = np.array([])
    label_list = np.array([])
    for sample in dataloader:
        y_predict = model(sample[0])
        y_predict = y_predict.detach().cpu().numpy().flatten()  # 展平输出
        predict_list = np.concatenate((predict_list, y_predict), axis=0)
        label_list = np.concatenate((label_list, sample[1].detach().cpu().numpy().flatten()), axis=0)
    
    mse = mean_squared_error(label_list, predict_list)
    mae = mean_absolute_error(label_list, predict_list)
    r2 = r2_score(label_list, predict_list)
    return mse, mae, r2

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
        
        if not os.path.exists(Result_log_folder + dataset_name + '/'):
            os.makedirs(Result_log_folder + dataset_name + '/')
        Initial_model_path = Result_log_folder + dataset_name + '/' + dataset_name + 'initial_model'
        model_save_path = Result_log_folder + dataset_name + '/' + dataset_name + 'Best_model'
        
        self.Result_log_folder = Result_log_folder
        self.dataset_name = dataset_name        
        self.model_save_path = model_save_path
        self.Initial_model_path = Initial_model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.print_result_every_x_epoch = print_result_every_x_epoch
        self.minium_batch_size = minium_batch_size
        
        self.lr = lr if lr is not None else 0.001
        self.Model = None
    
    def get_model(self, model):
        self.Model = model.to(self.device)
        
    def fit(self, X_train, y_train, X_val, y_val):
        print('code is running on ', self.device)
        
        # 转换数据类型为float32
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).float()
        
        # 移动到设备
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        # 添加通道维度
        if len(X_train.shape) == 2:
            X_train = X_train.unsqueeze_(1)
            X_val = X_val.unsqueeze_(1)
        
        # 保存初始权重
        torch.save(self.Model.state_dict(), self.Initial_model_path)
        
        # 使用MSE损失函数
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.Model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)
        
        # 创建DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(int(min(X_train.shape[0] / 10, self.batch_size)), self.minium_batch_size),
            shuffle=True
        )
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(int(min(X_val.shape[0] / 10, self.batch_size)), self.minium_batch_size),
            shuffle=False
        )
        
        self.Model.train()
        best_loss = float('inf')
        
        # 记录总训练时间
        total_start_time = time.time()
        
        for epoch in range(self.max_epoch):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.Model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            
            # 计算平均epoch loss
            epoch_loss = epoch_loss / len(train_loader.dataset)
            scheduler.step(epoch_loss)
            
            # 评估模型
            if eval_condition(epoch, self.print_result_every_x_epoch):
                # 打印学习率
                for param_group in optimizer.param_groups:
                    print(f'Epoch [{epoch+1}/{self.max_epoch}] lr: {param_group["lr"]:.6f}')
                
                # 评估模式
                self.Model.eval()
                with torch.no_grad():
                    train_mse, train_mae, train_r2 = eval_model(self.Model, train_loader)
                    val_mse, val_mae, val_r2 = eval_model(self.Model, val_loader)
                
                # 保存最佳模型
                if val_mse < best_loss:
                    best_loss = val_mse
                    torch.save(self.Model.state_dict(), self.model_save_path)
                
                # 计算当前周期用时
                epoch_time = time.time() - epoch_start_time
                
                # 打印日志
                log_message = (
                    f"Epoch {epoch+1}/{self.max_epoch}\t"
                    f"Train MSE: {train_mse:.4f} MAE: {train_mae:.4f} R2: {train_r2:.4f}\t"
                    f"Val MSE: {val_mse:.4f} MAE: {val_mae:.4f} R2: {val_r2:.4f}\t"
                    f"Epoch Time: {epoch_time:.2f}s"
                )
                print(log_message)
                save_to_log(log_message, self.Result_log_folder, self.dataset_name)
                
                self.Model.train()
        
        # 计算总训练时间
        total_time = time.time() - total_start_time
        print(f"Training complete. Best Val MSE: {best_loss:.4f}, Total Training Time: {total_time:.2f}s")
        
    def predict(self, X_test):
        self.Model.eval()
        X_test = torch.from_numpy(X_test).float().to(self.device)
        
        if len(X_test.shape) == 2:
            X_test = X_test.unsqueeze_(1)
        
        with torch.no_grad():
            predictions = self.Model(X_test).cpu().numpy()
        
        return predictions.flatten()  # 展平输出
    
    def load_model(self, model_path):
        self.Model.load_state_dict(torch.load(model_path))