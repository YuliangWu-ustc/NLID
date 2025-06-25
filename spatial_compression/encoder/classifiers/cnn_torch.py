import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os

from utils.utils import save_logs, save_logs_torch
from utils.utils import calculate_metrics

class CNN(nn.Module):
    """
    PyTorch实现的CNN分类模型
    """
    def __init__(self, input_shape, nb_classes):
        super(CNN, self).__init__()
        time_steps = input_shape[0]
        in_channels = input_shape[1]
        
        # 动态确定padding方式
        self.padding = nn.Identity()  # 默认无padding
        if time_steps < 60:
            self.padding = self._create_padding_layer(7)  # 第一层卷积的padding
            
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            self.padding,
            nn.Conv1d(in_channels, 6, kernel_size=7, padding=0),
            nn.Sigmoid(),
            nn.AvgPool1d(kernel_size=3, stride=3)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(6, 12, kernel_size=7, padding=0),
            nn.Sigmoid(),
            nn.AvgPool1d(kernel_size=3, stride=3)
        )
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self._get_flatten_size(input_shape), nb_classes)
        self.output_act = nn.Sigmoid()

    def _create_padding_layer(self, kernel_size):
        """创建same卷积所需的padding层"""
        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return nn.ConstantPad1d((pad_left, pad_right), 0)
    
    def _get_flatten_size(self, input_shape):
        """计算展平后的特征维度"""
        x = torch.randn(1, *input_shape)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.flatten(x).shape[1]

    def forward(self, x):
        # 调整输入维度 [batch, time_steps, features] -> [batch, features, time_steps]
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.output_act(x)

class Classifier_CNN_Torch:
    """
    CNN分类器封装类
    """
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        self.output_directory = output_directory
        self.nb_classes = nb_classes
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if build:
            self.model = CNN(input_shape, nb_classes).to(self.device)
            if verbose:
                print(self.model)
            # 保存初始化权重
            torch.save(self.model.state_dict(), os.path.join(output_directory, 'model_init.pth'))
    
    def fit(self, x_train, y_train, x_val, y_val, y_true, nb_epochs=2000, y_labels_description=None):
        # 数据预处理
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        y_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true

        # 创建数据集
        train_dataset = TensorDataset(
            torch.FloatTensor(x_train).to(self.device),
            y_train
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(x_val).to(self.device),
            y_val
        )

        # 数据加载器
        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 初始化优化器和损失函数
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()

        # 训练参数
        best_loss = np.inf
        history = {'loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        start_time = time.time()

        # hist = {}
        # hist['train_loss'] = []
        # hist['train_accuracy'] = []
        # hist['val_loss'] = []
        # hist['val_accuracy'] = []
        # hist['lr'] = []
        
        # hist['best_model'] = None
        # hist['best_model_train_loss'] = float('inf')
        # hist['best_model_val_loss'] = float('inf')
        # hist['best_model_train_accuracy'] = 0
        # hist['best_model_val_accuracy'] = 0
        
        for epoch in range(nb_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                
                # 计算训练准确率
                _, predicted = torch.max(outputs.data, 1)
                _, targets_idx = torch.max(targets, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets_idx).sum().item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    val_loss += criterion(outputs, targets).item() * inputs.size(0)
                    
                    # 计算验证准确率
                    _, predicted = torch.max(outputs.data, 1)
                    _, targets_idx = torch.max(targets, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets_idx).sum().item()
            
            # 计算平均损失和准确率
            train_loss /= len(train_dataset)
            val_loss /= len(val_dataset)
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            # 记录损失和准确率
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            # hist['train_loss'].append(train_loss)
            # hist['val_loss'].append(val_loss)
            # hist['train_accuracy'].append(train_accuracy)
            # hist['val_accuracy'].append(val_accuracy)
            # hist['lr'].append(optimizer.param_groups[0]['lr'])
            

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), 
                          os.path.join(self.output_directory, 'best_model.pth'))
                # hist['best_model'] = self.model.state_dict()
                # hist['best_model_train_loss'] = train_loss
                # hist['best_model_train_accuracy'] = train_accuracy
                # hist['best_model_val_accuracy'] = val_accuracy
            
            # 打印训练信息
            if self.verbose and epoch % 100 == 0:
                print(f'Epoch {epoch+1}/{nb_epochs} - Loss: {train_loss:.4f} '
                      f'Val Loss: {val_loss:.4f} - Train Acc: {train_accuracy:.4f} '
                      f'Val Acc: {val_accuracy:.4f}')

        # 保存最终模型
        torch.save(self.model.state_dict(), 
                  os.path.join(self.output_directory, 'last_model.pth'))
        
        # 加载最佳模型进行预测
        self.model.load_state_dict(torch.load(os.path.join(self.output_directory, 'best_model.pth')))
        self.model.eval()
        
        # 生成预测结果
        val_preds = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                outputs = self.model(inputs)
                val_preds.append(outputs.argmax(dim=1).cpu().numpy())
        y_pred = np.concatenate(val_preds)
        
        # 保存日志和指标
        duration = time.time() - start_time
        save_logs_torch(self.output_directory, history, duration, y_true, y_pred, y_labels_description=y_labels_description)
        # save_logs(self.output_directory, history, y_pred, y_true, duration, lr=False)

    def predict(self, x_test, y_true, return_df_metrics=True):
        # 加载最佳模型
        self.model.load_state_dict(torch.load(os.path.join(self.output_directory, 'best_model.pth')))
        self.model.eval()
        
        # 创建测试数据集
        test_dataset = TensorDataset(torch.FloatTensor(x_test).to(self.device))
        test_loader = DataLoader(test_dataset, batch_size=128)
        
        # 生成预测
        test_preds = []
        with torch.no_grad():
            for inputs in test_loader:
                outputs = self.model(inputs[0])
                test_preds.append(outputs.argmax(dim=1).cpu().numpy())
        y_pred = np.concatenate(test_preds)
        
        # 计算指标
        if return_df_metrics:
            y_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
            return calculate_metrics(y_true, y_pred, 0.0)
        return y_pred