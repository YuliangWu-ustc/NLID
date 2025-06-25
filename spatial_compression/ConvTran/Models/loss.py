import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module():
    """
    Get the loss module for model training.

    Returns:
        NoFussCrossEntropyLoss: An instance of the custom cross-entropy loss function with reduction='none'.
    """
    # return NoFussCrossEntropyLoss(reduction='none')  # ⭐ Returns the custom loss function instance
    return NoFussCrossEntropyLoss(reduction='none')  # ⭐ Returns the custom loss function instance


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))  # ⭐ 计算输出层权重的平方和作为L2正则项


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements

    改进版的交叉熵损失函数，放宽输入格式限制：
    - 自动将target转换为Long类型
    - 支持更灵活的target维度

    Args:
        inp (Tensor): 模型预测输出（通常是logits）
        target (Tensor): 真实标签（自动转换为long类型）
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=self.weight,  # ⭐ 核心操作：执行带类型转换的交叉熵计算
                               ignore_index=self.ignore_index, reduction=self.reduction)
