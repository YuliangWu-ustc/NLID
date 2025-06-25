import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import save_logs_torch, save_logs_torch_regr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PReLUShared(nn.Module):
    """实现时间维度参数共享的PReLU"""
    def __init__(self, channels):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, channels, 1) * 0.25)
        
    def forward(self, x):
        return torch.maximum(self.a * x, x)

class AttentionBlock(nn.Module):
    """自定义注意力机制模块"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        attention_data = x[:, :, :256]
        attention_softmax = x[:, :, 256:]
        attention_softmax = torch.softmax(attention_softmax, dim=-1)
        return attention_softmax * attention_data

class CNNLSTMModel(nn.Module):
    def __init__(self, input_shape, output_size=1):  # 修改为回归输出
        super().__init__()
        seq_len, n_features = input_shape
        
        # 卷积块1
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 128, 5, padding='same'),
            nn.InstanceNorm1d(128),
            PReLUShared(128),
            nn.Dropout(0.2),
            nn.MaxPool1d(2)
        )
        
        # 卷积块2
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 11, padding='same'),
            nn.InstanceNorm1d(256),
            PReLUShared(256),
            nn.Dropout(0.2),
            nn.MaxPool1d(2)
        )
        
        # 卷积块3
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, 21, padding='same'),
            nn.InstanceNorm1d(512),
            PReLUShared(512),
            nn.Dropout(0.2)
        )
        
        self.attention = AttentionBlock()
        
        self.dense = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.InstanceNorm1d(256)
        )
        
        # 输出层改为回归结构
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (seq_len // 4), output_size)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention(x.permute(0, 2, 1))
        x = self.dense(x)
        x = self.output(x)
        return x

class Regressor_CNN_LSTM_Torch:
    def __init__(self, output_directory, input_shape, verbose=False, device='cuda'):
        self.output_directory = output_directory
        self.device = device
        self.model = CNNLSTMModel(input_shape).to(device)
        self.verbose = verbose
        torch.save(self.model.state_dict(), f'{output_directory}/model_init.pth')
        
    def fit(self, x_train, y_train, x_val, y_val, y_true, batch_size=12, nb_epochs=100, y_labels_description=None):
        # 使用浮点型标签
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(x_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        criterion = nn.MSELoss()  # MSE损失
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

        history = {
            'loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_maxae': [], 'val_maxae': [],
            'train_mse': [], 'val_mse': [],
            'train_r2': [], 'val_r2': []  # 新增Train和Val R²
        }
        
        best_loss = float('inf')
        start_time = time.time()
        for epoch in range(nb_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            train_maxae = 0.0  # 直接记录最大值，无需平均
            train_mse = 0.0
            all_train_outputs = []  # 新增：收集所有预测值和真实值
            all_train_labels = []
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).view(-1, 1)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                # 计算指标
                mae = nn.L1Loss()(outputs, y_batch)
                batch_maxae = torch.max(torch.abs(outputs - y_batch)).item()
                mse = nn.MSELoss()(outputs, y_batch).item()
                
                # 更新累计指标
                train_loss += loss.item() * X_batch.size(0)
                train_mae += mae.item() * X_batch.size(0)
                train_maxae = max(train_maxae, batch_maxae)  # 直接取最大值，无需累加
                train_mse += mse * X_batch.size(0)
                train_total += X_batch.size(0)
                
                # 收集所有预测值和真实值用于计算 R²
                all_train_outputs.append(outputs.detach().cpu().numpy())
                all_train_labels.append(y_batch.cpu().numpy())

            # 计算训练集 R²（全局计算）
            all_train_outputs = np.concatenate(all_train_outputs)
            all_train_labels = np.concatenate(all_train_labels)
            train_r2 = r2_score(all_train_labels, all_train_outputs)

            # 计算平均指标（MaxAE 不需要平均）
            train_loss /= train_total
            train_mae /= train_total
            train_mse /= train_total

            # 验证阶段（同理修正）
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            val_maxae = 0.0
            val_mse = 0.0
            all_val_outputs = []  # 新增：收集所有验证集预测值和真实值
            all_val_labels = []
            val_total = 0

            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device).view(-1, 1)
                    outputs = self.model(X_val)
                    
                    # 计算指标
                    val_loss += criterion(outputs, y_val).item() * X_val.size(0)
                    val_mae += nn.L1Loss()(outputs, y_val).item() * X_val.size(0)
                    batch_maxae = torch.max(torch.abs(outputs - y_val)).item()
                    val_maxae = max(val_maxae, batch_maxae)  # 直接取最大值
                    val_mse += nn.MSELoss()(outputs, y_val).item() * X_val.size(0)
                    val_total += X_val.size(0)
                    
                    # 收集验证集数据
                    all_val_outputs.append(outputs.detach().cpu().numpy())
                    all_val_labels.append(y_val.cpu().numpy())

            # 计算验证集 R²（全局计算）
            all_val_outputs = np.concatenate(all_val_outputs)
            all_val_labels = np.concatenate(all_val_labels)
            val_r2 = r2_score(all_val_labels, all_val_outputs)

            # 计算平均指标（MaxAE 不需要平均）
            val_loss /= val_total
            val_mae /= val_total
            val_mse /= val_total

            # 记录指标
            history['train_maxae'].append(train_maxae)  # 直接记录最大值
            history['val_maxae'].append(val_maxae)
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), f'{self.output_directory}/best_model.pth')
            
            if self.verbose:
                print(f'Epoch {epoch+1}/{nb_epochs} - Loss: {train_loss:.4f} '
                      f'Val Loss: {val_loss:.4f} - Train MAE: {train_mae:.4f} '
                      f'Val MAE: {val_mae:.4f} - Train MaxAE: {train_maxae:.4f} '
                      f'Val MaxAE: {val_maxae:.4f} - Train MSE: {train_mse:.4f} '
                      f'Val MSE: {val_mse:.4f} - Train R²: {train_r2:.4f} '  # 输出Train R²
                      f'Val R²: {val_r2:.4f}')  # 输出Val R²
        
        torch.save(self.model.state_dict(), f'{self.output_directory}/last_model.pth')
        self.model.load_state_dict(torch.load(f'{self.output_directory}/best_model.pth'))
        
        # 计算最终指标
        y_pred = self.predict(x_val)
        final_mse = mean_squared_error(y_true, y_pred)
        final_mae = mean_absolute_error(y_true, y_pred)
        final_r2 = r2_score(y_true, y_pred)
        final_maxae = np.max(np.abs(y_true - y_pred))
        
        print(f'Final MSE: {final_mse:.4f}, MAE: {final_mae:.4f}, R²: {final_r2:.4f}, MaxAE: {final_maxae:.4f}')
        
        duration = time.time() - start_time
        save_logs_torch_regr(self.output_directory, history, duration, y_true, y_pred, y_labels_description=y_labels_description)
        
    def predict(self, x_test):
        self.model.eval()
        test_dataset = TensorDataset(torch.Tensor(x_test))
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        predictions = []
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch[0].to(self.device)
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions).squeeze()