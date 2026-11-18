import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from prettytable import PrettyTable
import numpy as np
import logging
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import itertools


standard_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


def get_input_chans(ch_names):
    input_chans = [0]  # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=4, feat_dim=2, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        if len(x.shape) == 3:
            x = x.squeeze(dim=-1)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        distmat = distmat - 2 * x @ self.centers.t()

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def get_logger(exp_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    path = os.path.join(exp_dir, 'log.txt')
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if os.path.exists(path):
        os.remove(path)
    hdlr = logging.FileHandler(path)
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    hdlr.setFormatter(formatter)
    console.setFormatter(formatter)

    logger.addHandler(hdlr)
    logger.addHandler(console)

    return logger


def move2device(x, device):
    if isinstance(x, list):
        y = []
        for item in x:
            y.append(move2device(item, device))
    elif isinstance(x, dict):
        y = {}
        for k, v in x.items():
            y[k] = move2device(v, device)
    elif x is None:
        y = None
    else:
        y = x.to(device)
    return y


class StepLR2(MultiStepLR):
    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6,
                 warm_up=False,
                 warm_up_ep=0,
                 warm_up_lr=1e-4):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        self.warm_up = warm_up
        self.warm_up_ep = warm_up_ep
        self.warm_up_lr = warm_up_lr
        self.start = None
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()

        if self.last_epoch < self.warm_up_ep and self.warm_up:
            if isinstance(lr_candidate, list):
                if self.start is None:
                    self.start = lr_candidate[0]
                for i in range(len(lr_candidate)):
                    lr_candidate[i] = self.warm_up_lr
            else:
                if self.start is None:
                    self.start = lr_candidate
                lr_candidate = self.warm_up_lr

        if self.last_epoch > self.warm_up_ep and self.start is not None:
            if isinstance(lr_candidate, list):
                for i in range(len(lr_candidate)):
                    lr_candidate[i] = self.start
            else:
                lr_candidate = self.start
            self.start = None

        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n
        # print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])
        # print(table)
        return kappa

    def plot(self, save_path=None, title='', normalize=True):  # 绘制混淆矩阵
        cm = self.matrix

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            print(cm)
        else:
            print(cm)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(self.labels))
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)
        # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
        # plt.ylim(len(self.labels) - 0.5, -0.5)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.clf()


def center_loss(feature, label, lambdas):
    center = nn.Parameter(torch.randn(int(max(label).item() + 1), feature.shape[1]), requires_grad=True).cuda()
    # print(center.shape)  # torch.Size([2, 2])
    # print(label.shape)  # torch.Size([5])

    center_exp = center.index_select(dim=0, index=label.long())
    # print(center_exp.shape)  # torch.Size([5, 2])

    count = torch.histc(label, bins=int(max(label).item() + 1), min=0, max=int(max(label).item()))
    # print(count)  # tensor([3., 2.], device='cuda:0')

    count_exp = count.index_select(dim=0, index=label.long())
    # print(count_exp)  # tensor([3., 3., 2., 3., 2.], device='cuda:0')

    loss = lambdas / 2 * torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2), dim=1), count_exp))
    return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


class CrossEntropyLoss_CosineSim(nn.Module):
    def __init__(self, gamma=1):
        super(CrossEntropyLoss_CosineSim, self).__init__()
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cosine = nn.CosineEmbeddingLoss()

    def forward(self, inputs, targets,
                feature=None, extras=None):
        ce_loss = self.cross_entropy(inputs, targets)

        if feature is None or extras is None:
            cos_loss = 0
        else:
            run_idx = extras['run_idx']
            select_idx = [torch.where(torch.logical_and(run_idx != run_idx[i], targets == targets[i]))
                          for i in range(run_idx.shape[0])]

            select_feature_list = [feature[select_i] for select_i in select_idx]
            left_feature_list = [feature[i].unsqueeze(dim=0).repeat(f.shape[0], 1) for i, f in
                                 enumerate(select_feature_list)]

            select_feature = torch.cat(select_feature_list, dim=0)
            left_feature = torch.cat(left_feature_list, dim=0)

            loss_flag = torch.ones(left_feature.shape[0], device=inputs.device)
            cos_loss = self.cosine(left_feature, select_feature, loss_flag)
            cos_loss = cos_loss * (loss_flag.shape[0] / (loss_flag.shape[0] - run_idx.shape[0]))

        loss = ce_loss + self.gamma * cos_loss
        return loss


class EmbeddingRunningMean:
    def __init__(self, subject_id, erm_beta, mi_feature_len, n_channels, device):
        self.subject_id = subject_id
        self.erm_beta = erm_beta
        self.mi_feature_len = mi_feature_len
        self.n_channels = n_channels
        self.device = device
        self.subject_mean = {k: torch.zeros([n_channels, mi_feature_len], device=device)
                             for k in subject_id}
        self.running_count = {k: torch.ones([n_channels], device=device)
                              for k in subject_id}

    def update(self, subject_emb, input_id):
        """
            subject_emb : [batch_size, n_channels * time_len, emb_size]
            input_id : [batch_size]
            mask : [batch_size, n_channels, time_len, 1]
        """
        with torch.no_grad():
            subject_emb = subject_emb.reshape(subject_emb.shape[0], *mask.shape[1:3], -1)

            for subject_emb_i, input_id_i, mask_i in zip(subject_emb, input_id, mask):
                subject_emb_i = subject_emb_i.transpose(0, 1)
                input_id_i = input_id_i.item()
                mask_i = mask_i.transpose(0, 1)
                mask_idx = (mask_i == 1).squeeze(dim=-1)

                for subject_emb_j, mask_idx_j in zip(subject_emb_i, mask_idx):
                    self.subject_mean[input_id_i][mask_idx_j] = (
                            self.subject_mean[input_id_i][mask_idx_j] * self.erm_beta +
                            subject_emb_j[mask_idx_j] * (1 - self.erm_beta)
                    )
                    self.running_count[input_id_i] = self.running_count[input_id_i] + mask_idx_j

    def get_subject_mean(self, subject_id):
        with torch.no_grad():
            return [self.subject_mean[idx] / (1 - self.erm_beta ** self.running_count[idx].unsqueeze(dim=-1))
                    for idx in subject_id]


def labram_load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=4, epsilon=0.05):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def exponential_moving_standardize(
        data, factor_new=0.001, init_block_size=None, eps=1e-4
):
    # From braindecode
    r"""Perform exponential moving standardization.

    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential moving variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{->v_t}, eps)`.


    Parameters
    ----------
    data: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: np.ndarray (n_channels, n_times)
        Standardized data.
    """
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_block_standardized = (data[0:init_block_size] - init_mean) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized.T

