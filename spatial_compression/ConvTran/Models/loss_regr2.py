import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(loss_name='MSELoss'):
    """
    根据损失函数名称获取对应的损失函数实例

    Args:
        loss_name (str): 损失函数名称，默认为 'MSELoss'

    Returns:
        nn.Module: 对应的损失函数实例
    """
    if loss_name == 'MSELoss':
        return NoFussMSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))  # ⭐ 计算输出层权重的平方和作为L2正则项


class NoFussMSELoss(nn.MSELoss):
    """
    改进版的均方误差损失函数，放宽输入格式限制：
    - 自动将target转换为Float类型
    - 支持更灵活的target维度

    Args:
        inp (Tensor): 模型预测输出（通常是连续值）
        target (Tensor): 真实标签（自动转换为float类型）
    """

    def forward(self, inp, target):
        # print(f'in NoFussMSELoss, inp.shape: {inp.shape}')
        # print(f'in NoFussMSELoss, target.shape: {target.shape}')
        return F.mse_loss(inp, target.float(), reduction='none')  # 使用均方误差损失
