import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from utils.utils import save_logs_torch
from utils.utils import calculate_metrics

class Classifier_FCN_Torch:
    """
    实现一个全卷积网络(FCN)分类器，用于时间序列分类任务，基于PyTorch实现
    
    Attributes:
        output_directory (str): 模型输出目录路径
        input_shape (tuple): 输入数据的形状
        nb_classes (int): 分类类别数量
        verbose (bool): 是否显示模型摘要信息
        model (nn.Module): 构建的FCN模型实例
    """

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        """
        初始化FCN分类器
        
        Args:
            output_directory (str): 保存模型和权重的目录路径
            input_shape (tuple): 输入数据的形状(如时间步长,特征维度)
            nb_classes (int): 分类的类别数量
            verbose (bool, optional): 是否打印模型摘要. Defaults to False.
            build (bool, optional): 是否立即构建模型. Defaults to True.
            y_labels_description (list, optional): 标签描述. Defaults to None.
        """
        self.output_directory = output_directory
        if build:
            self.model = self.build_model(input_shape, nb_classes)  # ⭐ 构建FCN模型结构
            if verbose:
                print(self.model)
            self.verbose = verbose
            torch.save(self.model.state_dict(), self.output_directory + 'model_init.pth')  # ⭐ 保存初始权重
        return
    
    def build_model(self, input_shape, nb_classes):
        class FCN(nn.Module):
            def __init__(self, input_shape, nb_classes):
                super(FCN, self).__init__()
                self.conv1 = nn.Conv1d(input_shape[0], 128, kernel_size=8, padding='same')  # ⭐ 修改为input_shape[0]
                self.bn1 = nn.BatchNorm1d(128)
                self.relu1 = nn.ReLU()
                
                self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding='same')
                self.bn2 = nn.BatchNorm1d(256)
                self.relu2 = nn.ReLU()
                
                self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding='same')
                self.bn3 = nn.BatchNorm1d(128)
                self.relu3 = nn.ReLU()
                
                self.gap = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(128, nb_classes)
                self.softmax = nn.Softmax(dim=1)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu2(x)
                
                x = self.conv3(x)
                x = self.bn3(x)
                x = self.relu3(x)
                
                x = self.gap(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                x = self.softmax(x)
                return x
        
        model = FCN(input_shape, nb_classes)
        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, nb_epochs=2000, print_every=10, y_labels_description=None):
        if not torch.cuda.is_available():
            print('error')
            exit()
        
        batch_size = 16
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()

        hist = {}
        hist['train_loss'] = []
        hist['train_accuracy'] = []
        # hist['val_loss'] = []
        hist['val_accuracy'] = []
        hist['lr'] = []
        
        hist['best_model'] = None
        hist['best_model_train_loss'] = float('inf')
        hist['best_model_train_accuracy'] = 0
        hist['best_model_val_accuracy'] = 0
        
        for epoch in range(nb_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for i in range(0, x_train.shape[0], mini_batch_size):
                x_batch = torch.tensor(x_train[i:i + mini_batch_size], dtype=torch.float32)
                y_batch = torch.tensor(y_train[i:i + mini_batch_size], dtype=torch.float64)  # ⭐ 确保y_batch是Long类型
                
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                # 计算当前batch的准确率
                _, predicted = torch.max(outputs.data, 1)
                y_batch_argmax = torch.argmax(y_batch, dim=1)
                epoch_correct += (predicted == y_batch_argmax).sum().item()
                epoch_total += y_batch.size(0)
                epoch_loss += loss.item()
            
            # 计算整个epoch的平均损失和准确率
            epoch_loss /= (x_train.shape[0] / mini_batch_size)
            epoch_accuracy = epoch_correct / epoch_total
            
            hist['train_loss'].append(epoch_loss)
            hist['train_accuracy'].append(epoch_accuracy)
            
            # 计算验证集的准确率
            self.model.eval()
            with torch.no_grad():
                y_pred_val = self.model(torch.tensor(x_val, dtype=torch.float32))
                y_pred_val = torch.argmax(y_pred_val, dim=1).numpy()
                y_val_argmax = torch.argmax(torch.tensor(y_val), dim=1).numpy()
                val_accuracy = np.sum(y_pred_val == y_val_argmax) / len(y_val_argmax)
                hist['val_accuracy'].append(val_accuracy)
                hist['lr'].append(optimizer.param_groups[0]['lr'])

                if val_accuracy > hist['best_model_val_accuracy']:
                    hist['best_model_val_accuracy'] = val_accuracy
                    hist['best_model_val_accuracy'] = val_accuracy
                    hist['best_model'] = self.model.state_dict()
                    hist['best_model_train_loss'] = epoch_loss
                    hist['best_model_train_accuracy'] = epoch_accuracy
                
            if self.verbose and (epoch + 1) % print_every == 0:
                print(f'Epoch {epoch + 1}/{nb_epochs}, Train Loss: {epoch_loss}, Train Accuracy: {epoch_accuracy}, Val Accuracy: {val_accuracy}')

        duration = time.time() - start_time

        # torch.save(self.model.state_dict(), self.output_directory + 'last_model.pth')
        torch.save(hist['best_model'], self.output_directory + 'best_model.pth')

        self.model.load_state_dict(torch.load(self.output_directory + 'best_model.pth'))
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.tensor(x_val, dtype=torch.float32))
            y_pred = torch.argmax(y_pred, dim=1).numpy()

        save_logs_torch(self.output_directory, hist, duration, y_true, y_pred, y_labels_description=y_labels_description)

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        self.model.load_state_dict(torch.load(self.output_directory + 'last_model.pth'))
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.tensor(x_test, dtype=torch.float32))
            if return_df_metrics:
                y_pred = torch.argmax(y_pred, dim=1).numpy()
                df_metrics = calculate_metrics(y_true, y_pred, 0.0)
                return df_metrics
            else:
                return y_pred.numpy() 