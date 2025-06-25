import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import save_logs_torch
from sklearn.metrics import f1_score  # 导入 f1_score

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
        # 分割特征
        attention_data = x[:, :, :256]
        attention_softmax = x[:, :, 256:]
        # 应用Softmax
        attention_softmax = torch.softmax(attention_softmax, dim=-1)
        # 相乘并返回
        return attention_softmax * attention_data

class CNNLSTMModel(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super().__init__()
        seq_len, n_features = input_shape  # 输入形状 (batch, seq_len, features)
        
        # 卷积块1
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 128, 5, padding='same'),  # 输入通道改为n_features
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
        
        # 注意力机制
        self.attention = AttentionBlock()
        
        # 全连接层
        self.dense = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.InstanceNorm1d(256)
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (seq_len // 4), nb_classes),  # 确保输入形状匹配
            # nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # 调整输入维度 (batch, seq_len, features) -> (batch, features, seq_len)
        # print(f'Input shape: {x.shape}')  # 应该是 (batch_size, seq_len, n_features)
        x = x.permute(0, 2, 1)
        # print(f'After permute: {x.shape}')  # 应该是 (batch_size, n_features, seq_len)
        
        x = self.conv1(x)
        # print(f'After conv1: {x.shape}')
        
        x = self.conv2(x)
        # print(f'After conv2: {x.shape}')
        
        x = self.conv3(x)
        # print(f'After conv3: {x.shape}')
        
        # 应用注意力
        x = self.attention(x.permute(0, 2, 1))  # 转回 (batch, seq_len, features)
        # print(f'After attention: {x.shape}')
        
        x = self.dense(x)
        # print(f'After dense: {x.shape}')
        
        x = self.output(x)
        # print(f'After output: {x.shape}')
        
        return x

class Classifier_CNN_LSTM_Torch:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, device='cuda'):
        self.output_directory = output_directory
        self.device = device
        self.model = CNNLSTMModel(input_shape, nb_classes).to(device)
        self.verbose = verbose
        
        # 保存初始化权重
        torch.save(self.model.state_dict(), f'{output_directory}/model_init.pth')
        
    def fit(self, x_train, y_train, x_val, y_val, y_true, batch_size=12, nb_epochs=100, y_labels_description=None):
        # 转换数据为PyTorch张量
        # print(1)
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(x_val), torch.LongTensor(y_val))
        # print(2)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # print(3)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        
        # 训练参数
        history = {'loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [], 'train_f1': [], 'val_f1': []}
        best_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(nb_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_preds = []
            train_labels = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                # print(4)
                outputs = self.model(X_batch)
                # print(5)
                # raise Exception('stop')
                
                # 调试信息
                # print(f'outputs shape: {outputs.shape}')  # 应该是 (batch_size, num_classes)
                # print(f'y_batch shape: {y_batch.shape}')  # 应该是 (batch_size)
                # print(f'outputs dtype: {outputs.dtype}')  # 应该是 float32
                # print(f'y_batch dtype: {y_batch.dtype}')  # 应该是 int64 (Long)
                # print(f'outputs: {outputs}')
                # 将y_batch转换成float
                y_batch = y_batch.float()
                
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
                # 计算训练准确率
                predicted = torch.argmax(outputs.data, dim=1)
                y_batch_id = y_batch.argmax(dim=1)
                # print(f'predicted: {predicted}')
                # print(f'y_batch_id: {y_batch_id}')
                # print(f'predicted.shape: {predicted.shape}')
                # print(f'y_batch_id.shape: {y_batch_id.shape}')
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch_id).sum().item()
                
                # 收集训练预测和标签
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(y_batch_id.cpu().numpy())
            
            # 计算训练 F1 Score
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    outputs = self.model(X_val)
                    y_val = y_val.float()
                    val_loss += criterion(outputs, y_val).item() * X_val.size(0)
                    
                    # 计算验证准确率
                    _, predicted = torch.max(outputs.data, 1)
                    y_val_id = y_val.argmax(dim=1)
                    val_total += y_val.size(0)
                    val_correct += (predicted == y_val_id).sum().item()
                    
                    # 收集验证预测和标签
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(y_val_id.cpu().numpy())
            
            # 计算验证 F1 Score
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            
            # 计算平均损失和准确率
            train_loss /= len(train_dataset)
            val_loss /= len(val_dataset)
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            # 记录损失、准确率和 F1 Score
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), f'{self.output_directory}/best_model.pth')
            
            # 打印训练信息
            if self.verbose:
                print(f'Epoch {epoch+1}/{nb_epochs} - Loss: {train_loss:.4f} '
                      f'Val Loss: {val_loss:.4f} - Train Acc: {train_accuracy:.4f} '
                      f'Val Acc: {val_accuracy:.4f} - Train F1: {train_f1:.4f} '
                      f'Val F1: {val_f1:.4f}')
        
        # 保存最终模型
        torch.save(self.model.state_dict(), f'{self.output_directory}/last_model.pth')
        
        # 加载最佳模型进行预测
        self.model.load_state_dict(torch.load(f'{self.output_directory}/best_model.pth'))
        self.model.eval()
        
        # 生成预测结果
        val_preds = []
        with torch.no_grad():
            for X_val, _ in val_loader:
                X_val = X_val.to(self.device)
                outputs = self.model(X_val)
                val_preds.append(outputs.argmax(dim=1).cpu().numpy())
        y_pred = np.concatenate(val_preds)
        
        # 计算最终 F1 Score
        final_f1 = f1_score(y_true, y_pred, average='weighted')
        print(f'Final F1 Score: {final_f1:.4f}')
        
        # 保存日志和指标
        duration = time.time() - start_time
        save_logs_torch(self.output_directory, history, duration, y_true, y_pred, y_labels_description=y_labels_description)
        
    def predict(self, x_test):
        # 加载最佳模型
        self.model.load_state_dict(torch.load(f'{self.output_directory}/best_model.pth'))
        self.model.eval()
        
        test_dataset = TensorDataset(torch.Tensor(x_test))
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        predictions = []
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch[0].to(self.device)
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())
        
        return np.argmax(np.concatenate(predictions), axis=1)