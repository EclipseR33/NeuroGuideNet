import numpy as np
import torch
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, label):
        for t in self.transforms:
            if t is None:
                continue
            x, label = t(x, label)
        return x, label


class RandomCrop:
    def __init__(self, points=750):
        self.points = int(points)

    def __call__(self, x: torch.tensor, label):
        T = x.shape[-2]
        s = random.randint(0, T - self.points)
        return x[..., s:s + self.points, :], label


class RandomErasing:
    """
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased time duration wrt input EEG trials.
         max_area: Maximum percentage of erased time duration wrt input EEG trials.
    """

    def __init__(self, probability=0.5, min_area=0.02, max_area=1 / 3, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        self.device = device

    def _erase(self, x, width):
        if random.random() < self.probability:
            target_area = random.uniform(self.min_area, self.max_area) * width
            w = int(target_area)
            left = random.randint(0, width - w)
            x[..., left:left + w, :].zero_()

    def __call__(self, input, label):
        if len(input.size()) == 2:
            self._erase(input, *input.size())
        else:
            input_shape = input.shape
            for i in range(input_shape[0]):
                self._erase(input[i], input_shape[-1])
        return input, label


class RepeatedTrialAugmentation:
    def __init__(self, transform=torch.nn.Identity(), m=5):
        self.transform = transform
        self.m = m

    def __call__(self, x, y):
        if self.m == 0:
            return x, y
        if self.m == 1:
            return self.transform(x, y)

        data = torch.cat([self.transform(x.clone(), y) for _ in range(self.m)], dim=0)
        labels = torch.cat([y] * self.m, dim=0)
        return data, labels


class CutMix:
    def __init__(self, probability=0.5, min_area=0.02, max_area=1 / 3, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        self.device = device

    def _cutmix(self, x1, x2, width):
        if random.random() < self.probability:
            target_area = random.uniform(self.min_area, self.max_area) * width
            w = int(target_area)
            left = random.randint(0, width - w)
            x1[..., left:left + w, :] = x2[..., left:left + w, :]

    def __call__(self, input, in_label):
        width = input.size(-1)
        label = in_label.cpu()
        types = torch.unique(label).tolist()
        for c in types:
            idxes = torch.where(label == c)[0].tolist()
            for i in range(len(idxes) - 1):
                self._cutmix(input[idxes[i]], input[idxes[i + 1]], width)
        return input, in_label


class SegmentReassemble:
    def __init__(self, p1=0.5, split_num=3):
        self.p1 = p1
        self.split_num = split_num

    def s_r(self, x, idxes):
        if random.random() < self.p1:
            x_copy = x[idxes].clone()
            x_chunk = torch.chunk(x_copy, self.split_num, dim=-2)
            rand_idx = [random.sample(np.arange(x_copy.shape[0]).tolist(), x_copy.shape[0])
                        for i in range(self.split_num)]
            x_reassemble = [x_chunk[i][rand_idx[i]] for i in range(self.split_num)]
            x_new = torch.cat(x_reassemble, dim=2)
            x[idxes] = x_new

    def __call__(self, x, in_label):
        label = in_label.cpu()
        types = torch.unique(label).tolist()
        for c in types:
            idxes = torch.where(label == c)[0].tolist()
            self.s_r(x, idxes)
        return x, in_label


class AddNoise:
    def __init__(self, p1=0.5, p_ch=0.2, noise_max_scale=0.2, device='cuda'):
        self.p1 = p1
        self.p_ch = p_ch
        self.noise_max_scale = noise_max_scale
        self.device = device

    def __call__(self, input, label):
        if random.random() < self.p1:
            p_channel = torch.rand(input.shape[1],
                                   device=self.device) < self.p_ch
            noise_scale = torch.rand(input.shape[1],
                                     device=self.device) * self.noise_max_scale * p_channel
            noise_scale = noise_scale[None, :, None, None]
            noise = torch.randn_like(input, device=self.device)
            input = input * (1 - noise_scale) + noise * noise_scale
        return input, label


class RandomFlipBCIIV2a:
    def __init__(self, label_name, p1=0.3, device='cuda'):
        self.label_name = label_name
        self.p1 = p1
        self.device = device

        self.ch_name = [
            'FZ',
            'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
            'P1', 'PZ', 'P2',
            'POZ'
        ]
        self.left_name = [
            'FC3', 'FC1',
            'C5', 'C3', 'C1',
            'CP3', 'CP1',
            'P1'
        ]
        self.right_name = [
            'FC2', 'FC4',
            'C2', 'C4', 'C6',
            'CP2', 'CP4',
            'P2',
        ]
        self.left_idx = [self.ch_name.index(name) for name in self.left_name]
        self.right_idx = [self.ch_name.index(name) for name in self.right_name]
        self.left_label = self.label_name.index('left')
        self.right_label = self.label_name.index('right')

    def __call__(self, x, label):
        if random.random() < self.p1:
            x_left = x[:, self.left_idx]
            x[:, self.left_idx] = x[:, self.right_idx]
            x[:, self.right_idx] = x_left

            label[label == self.left_label] = -1
            label[label == self.right_label] = -2
            label[label == -1] = self.right_label
            label[label == -2] = self.left_label
        return x, label


class RandomFlipOpenBMI:
    def __init__(self, label_name, p1=0.3, device='cuda'):
        self.label_name = label_name
        self.p1 = p1
        self.device = device

        self.ch_name = [
            'Fp1', 'Fp2',
            'F7', 'F3', 'Fz', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'PO9', 'O1', 'Oz', 'O2', 'PO10',
            'FC3', 'FC4',
            'C5', 'C1', 'C2', 'C6',
            'CP3', 'CPz', 'CP4',
            'P1', 'P2',
            'POz',
            'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h',
            'TPP8h', 'TP8', 'TPP10h',
            'F9', 'F10',
            'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'
        ]
        self.left_name = [
            'Fp1', 'AF7', 'AF3', 'F9', 'F7', 'F3', 'FT9', 'FC5', 'FC3', 'FC1', 'FTT9h',
            'T7', 'C5', 'C3', 'C1', 'TTP7h', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'TPP9h',
            'P7', 'P3', 'P1', 'PO9', 'PO3', 'O1',
        ]
        self.right_name = [
            'Fp2', 'AF4', 'AF8', 'F4', 'F8', 'F10', 'FC2', 'FC4', 'FC6', 'FT10', 'FTT10h',
            'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'TPP8h', 'TPP10h',
            'P2', 'P4', 'P8', 'PO4', 'PO10', 'O2'
        ]
        self.left_idx = [self.ch_name.index(name) for name in self.left_name]
        self.right_idx = [self.ch_name.index(name) for name in self.right_name]
        self.left_label = self.label_name.index('left')
        self.right_label = self.label_name.index('right')

    def __call__(self, x, label):
        if random.random() < self.p1:
            x_left = x[:, self.left_idx]
            x[:, self.left_idx] = x[:, self.right_idx]
            x[:, self.right_idx] = x_left

            label[label == self.left_label] = -1
            label[label == self.right_label] = -2
            label[label == -1] = self.right_label
            label[label == -2] = self.left_label
        return x, label
