import os
import logging
import torch
import numpy as np
from collections import OrderedDict
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from Models.loss import l2_reg_loss
from Models import utils, analysis

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less


class BaseTrainer(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat =False):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class SupervisedTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):

        super(SupervisedTrainer, self).__init__(*args, **kwargs)

        if isinstance(args[3], torch.nn.CrossEntropyLoss):
            # self.classification = True  # True if classification, False if regression
            self.analyzer = analysis.Analyzer(print_conf_mat=False)
        else:
            self.classification = False
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):

            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))

            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            '''
            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)
            '''
            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True, return_pred_labels=False):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            per_batch['targets'].append(targets.cpu().numpy())
            predictions = predictions.detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append([loss.cpu().numpy()])
            per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            #if i % self.print_interval == 0:
                #ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                #self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions,dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

        '''
        if self.model.num_classes == 2:
            false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets, probs[:, 1])  # 1D scores needed
            self.epoch_metrics['AUROC'] = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = sklearn.metrics.precision_recall_curve(targets, probs[:, 1])
            self.epoch_metrics['AUPRC'] = sklearn.metrics.auc(rec, prec)
        '''
        if return_pred_labels:
            return self.epoch_metrics, metrics_dict, predictions, targets
        else:
            return self.epoch_metrics, metrics_dict


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""
    """
    在验证集上执行模型评估并处理评估结果

    Args:
        val_evaluator: 验证集评估器，包含模型和验证数据集
        tensorboard_writer: TensorBoard写入器，用于记录评估指标
        config: 配置字典，包含关键指标等配置信息
        best_metrics: 当前最佳评估指标字典
        best_value: 当前关键指标的最佳值
        epoch: 当前训练周期数

    Returns:
        tuple: 包含三个元素的元组
            - aggr_metrics: 当前评估的聚合指标
            - best_metrics: 更新后的最佳指标(如果当前模型更好)
            - best_value: 更新后的关键指标最佳值(如果当前模型更好)
    """
    with torch.no_grad():
        aggr_metrics, ConfMat = val_evaluator.evaluate(epoch, keep_all=True)  # ⭐ 核心评估操作：在验证集上执行模型评估

    print()
    print_str = 'Validation Summary: '
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)  # ⭐ 核心判断：比较当前关键指标与历史最佳值
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

    return aggr_metrics, best_metrics, best_value


def train_runner(config, model, trainer, val_evaluator, path):
    """
    执行模型训练的主循环，包括训练、验证、指标记录和最佳模型保存。

    Args:
        config (dict): 包含训练配置参数的字典（如epochs, optimizer等）
        model: 待训练的模型实例
        trainer: 训练器对象，负责执行训练过程
        val_evaluator: 验证器对象，负责执行验证过程
        path (str): 模型保存路径

    Returns:
        None
    """
    epochs = config['epochs']
    optimizer = config['optimizer']
    loss_module = config['loss_module']
    start_epoch = 0
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    best_value = 1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    save_best_model = utils.SaveBestModel()
    # save_best_acc_model = utils.SaveBestACCModel()

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # ⭐ 执行当前epoch的训练并获取训练指标
        aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                              best_value, epoch)
        save_best_model(aggr_metrics_val['loss'], epoch, model, optimizer, loss_module, path)
        # save_best_acc_model(aggr_metrics_val['accuracy'], epoch, model, optimizer, loss_module, path)

        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))

        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return