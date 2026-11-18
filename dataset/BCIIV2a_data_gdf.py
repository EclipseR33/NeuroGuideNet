import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
import mne
import glob
import os
import math
import scipy.io as scio
from scipy import signal
from scipy.linalg import sqrtm, inv
import pywt

from utils import exponential_moving_standardize


class BCIIV2ADataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

        self.ch_names = ['FZ',
                         'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
                         'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                         'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
                         'P1', 'PZ', 'P2',
                         'POZ']
        map_idx = [[0, 3],
                   [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
                   [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
                   [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
                   [4, 2], [4, 3], [4, 4],
                   [5, 3]]

        self.map_idx = (torch.LongTensor([i[0] for i in map_idx]),
                        torch.LongTensor([i[1] for i in map_idx]))

        self.ch_idx = torch.LongTensor([
            19,
            28, 29, 30, 31, 32,
            38, 39, 40, 41, 42, 43, 44,
            50, 51, 52, 53, 54,
            62, 63, 64,
            74
        ]) + 1
        self.ch_idx = torch.concatenate([torch.LongTensor([0]), self.ch_idx])

    def __len__(self):
        return len(self.data_dict['data'])

    def __getitem__(self, idx):
        subject_id = int(self.data_dict['subject_id'][idx].split('A')[-1])
        run_idx = self.data_dict['run_idx'][idx]
        data = self.data_dict['data'][idx]
        label = self.data_dict['label'][idx]

        data = torch.as_tensor(data, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.long)
        subject_id = torch.as_tensor(subject_id, dtype=torch.long)
        return data, label, subject_id


def extract_events(events_input, raw, filtered_data, picks,
                   data_len, window_size, stride, is_test=False):
    """
    :param events:
    :return:
    Event type
    276 Idling EEG (eyes open)
    277 Idling EEG (eyes closed)
    768 Start of a trial
    769 Cue onset left (class 1) - label 0
    770 Cue onset right (class 2) - label 1
    771 Cue onset foot (class 3) - label 2
    772 Cue onset tongue (class 4) - label 3
    783 Cue unknown
    1023 Rejected trial
    1072 Eye movements
    32766 Start of a new run
    events_key {'1023': 1, '1072': 2, '276': 3, '277': 4, '32766': 5,
                '768': 6, '769': 7, '770': 8, '771': 9, '772': 10}
    """

    events_array = events_input[0]
    events_key = events_input[1]
    # print('events_array', events_array)
    print('events_key', events_key)

    try:
        class2label = {
            events_key['769']: 0, events_key['770']: 1,
            events_key['771']: 2, events_key['772']: 3,
        }
    except:
        class2label = {events_key['783']: -1}

    data_dict = {'cue_idx': [], 'run_idx': [], 'data': [], 'label': []}
    run_idx_count = -1

    # pre_len = 1.8

    cue_cnt = 0
    for i, event in enumerate(events_array):
        if event[-1] == events_key['32766']:
            run_idx_count += 1
        elif event[-1] == events_key['768']:  # start of trail
            same_idx = events_array[events_array[:, 0] == event[0]]
            if np.any(same_idx[:, -1] == events_key['1023']):  # rejected trail
                # print('rejected')
                continue
        elif event[-1] in class2label.keys():

            # 2s - cue start | 3s cue end & imagery start | 6s imagery end | break
            # start_idx = event[0] + raw.time_as_index([1])[0]
            start_idx = event[0]

            if is_test:
                end_idx = start_idx + raw.time_as_index([round((data_len) * 1e5) / 1e5])[0]
                # data, time = raw[picks, start_idx:end_idx]

                data = filtered_data[picks, start_idx:end_idx]

                label = class2label[event[-1]]

                if len(data.shape) == 2:
                    data = data[:, :, None]
                assert len(data.shape) == 3
                assert data.shape[1] == raw.time_as_index([data_len])[0], \
                    (f'data len : {data.shape[1]} | event[0] {event[0]} | start_idx {start_idx} | end_idx {end_idx} | '
                     f'raw.time_as_index(data_len)[0] {raw.time_as_index([data_len])[0]}')

                # plt.figure()
                # plt.plot(data[0, :, 0].T)
                # plt.title("Sample channels")
                # plt.show()
                # plt.figure()
                # plt.plot(data[0, :, 1].T)
                # plt.title("Sample channels")
                # plt.show()

                # plt.figure()
                # plt.contourf(np.arange(data[0].shape[0]),
                #              np.arange(data[0].shape[1]),
                #              abs(data[0].T))
                # plt.subplots_adjust(hspace=0.4)
                # plt.show()

                data_dict['cue_idx'].append(cue_cnt)
                data_dict['run_idx'].append(run_idx_count)
                data_dict['data'].append(data)
                data_dict['label'].append(label)
            else:
                i = 0
                while i + window_size <= data_len:
                    # print(start_idx + i, start_idx + i + window_size)

                    line = raw.time_as_index([round((i) * 1e5) / 1e5,
                                              round((i + window_size) * 1e5) / 1e5])
                    # data, time = raw[picks, start_idx + line[0]:
                    #                         start_idx + line[1]]

                    data = filtered_data[picks, start_idx + line[0]:start_idx + line[1]]

                    label = class2label[event[-1]]

                    if len(data.shape) == 2:
                        data = data[:, :, None]
                    assert len(data.shape) == 3
                    assert data.shape[1] == raw.time_as_index([window_size])[0], f'data len : {data.shape[1]}'

                    # plt.figure()
                    # plt.plot(data[0, :, 0].T)
                    # plt.title("Sample channels")
                    # plt.show()

                    data_dict['cue_idx'].append(cue_cnt)
                    data_dict['run_idx'].append(run_idx_count)
                    data_dict['data'].append(data)
                    data_dict['label'].append(label)
                    i += stride

            cue_cnt += 1
    return data_dict


def read_gdf_data(root, path, filter_order,
                  data_len, window_size, stride, fs, f_start, f_end,
                  mat_label=None, factor_new=0.001,
                  return_scale_values=False):

    mapping = {
        'EEG-Fz': 'eeg', 'EEG-0': 'eeg', 'EEG-1': 'eeg', 'EEG-2': 'eeg', 'EEG-3': 'eeg', 'EEG-4': 'eeg', 'EEG-5': 'eeg',
        'EEG-C3': 'eeg', 'EEG-6': 'eeg', 'EEG-Cz': 'eeg', 'EEG-7': 'eeg', 'EEG-C4': 'eeg', 'EEG-8': 'eeg',
        'EEG-9': 'eeg', 'EEG-10': 'eeg', 'EEG-11': 'eeg', 'EEG-12': 'eeg', 'EEG-13': 'eeg', 'EEG-14': 'eeg',
        'EEG-Pz': 'eeg', 'EEG-15': 'eeg', 'EEG-16': 'eeg', 'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'
    }

    if return_scale_values:
        scale_values = {}

    all_data_dict = {'subject_id': [], 'run_idx': [], 'data': [], 'label': []}
    for p in path:
        # print('p', p)
        raw = mne.io.read_raw_gdf(p, preload=True)
        raw = raw.set_channel_types(mapping)

        raw.load_data()
        raw.set_eeg_reference(ref_channels='average')
        # raw.plot()

        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')[:22]
        raw = raw.pick(picks)
        if fs != raw.info['sfreq']:
            raw = raw.resample(fs)

        if not isinstance(f_start, list):
            filtered_data = raw.copy().filter(
                l_freq=f_start, h_freq=f_end, method='iir',
                iir_params={'order': filter_order, 'ftype': 'butter'}
            ).get_data() * 1e6

            filtered_data = exponential_moving_standardize(filtered_data,
                                                           factor_new=factor_new,
                                                           init_block_size=int(raw.info['sfreq'] * 4))
        else:
            filtered_data = []
            for low, high in zip(f_start, f_end):
                temp_data = raw.copy().filter(
                    l_freq=low, h_freq=high, method='iir',
                    iir_params={'order': filter_order, 'ftype': 'butter'}
                ).get_data() * 1e6

                temp_data = exponential_moving_standardize(temp_data,
                                                           factor_new=factor_new,
                                                           init_block_size=int(raw.info['sfreq'] * 4))
                filtered_data.append(temp_data)
            filtered_data = np.stack(filtered_data, -1)
        # raw.plot()
        # print('raw', raw)
        # print('raw info', raw.info)
        # print('raw ch_names', raw.ch_names)

        # raw.plot()
        # print('raw', raw)
        # print('raw info', raw.info)
        # print('raw ch_names', raw.ch_names)

        events = mne.events_from_annotations(raw)
        print('events', events)

        is_test = True if mat_label is not None else False
        subject_data_dict = extract_events(events, raw, filtered_data, picks,
                                           data_len, window_size, stride, is_test)
        all_data_dict['subject_id'] += [p.split(os.sep)[-1].split('.')[0][:-1]
                                        for i in range(len(subject_data_dict['run_idx']))]

        all_data_dict['run_idx'] += subject_data_dict['run_idx']
        all_data_dict['data'] += subject_data_dict['data']

        if mat_label is None:
            all_data_dict['label'] += subject_data_dict['label']
        else:
            file_name = p.split(os.sep)[-1].split('.')[0]
            label = scio.loadmat(
                str(
                    os.path.join(
                        os.path.join(root, mat_label),
                        file_name + '.mat'
                    )
                )
            )['classlabel']
            assert label.shape[0] == len(subject_data_dict['data'])
            label = label - 1
            all_data_dict['label'] += label.squeeze(-1).tolist()

        if return_scale_values:
            temp_data = np.array(all_data_dict['data'])
            temp_data = temp_data.transpose(1, 0, 2, 3).reshape(22, -1)
            scale_values[p.split(os.sep)[-1].split('.')[0]] = \
                [
                    temp_data.min(-1),
                    temp_data.max(-1),
                    np.percentile(temp_data, 10, axis=-1),
                    np.percentile(temp_data, 90, axis=-1)
                ]
    # print(all_data_dict['subject_id'])
    if return_scale_values:
        return {k: np.array(all_data_dict[k]) for k in all_data_dict}, scale_values
    else:
        return {k: np.array(all_data_dict[k]) for k in all_data_dict}


def check_scale():
    filter_order = 5
    data_len = 3
    window_size = 1
    stride = 0.5

    train_data_dict, scale_values = read_gdf_data(
        root_2a,
        '*.gdf',
        filter_order=filter_order,
        data_len=data_len,
        window_size=window_size,
        stride=stride,
        fs=250,
        f_start=0.5,
        f_end=45,
        # mat_label='true_labels/',
        return_scale_values=True
    )
    print(scale_values)

    for key in scale_values:
        plt.figure()
        plt.title(key)
        data = scale_values[key]
        x = np.arange(data[0].shape[0]) + 1
        plt.scatter(x, data[0], label='min')
        plt.scatter(x, data[1], label='max')
        plt.scatter(x, data[2], label='percentile 10')
        plt.scatter(x, data[3], label='percentile 90')
        plt.legend()
        plt.savefig(f'../img/data_scale/scale_{key}.png')


if __name__ == '__main__':
    from sklearn.model_selection import KFold

    # 4-class
    root_2a = r'F:\AI\Dataset\BCIC\BCICIV_2a_gdf'
    # train_val_data_dict = read_data(root_2a, '*T.gdf')
    # # test_data_dict = read_test(root_2a, '*E.gdf')
    # print('train & val len', len(train_val_data_dict['data']))
    #
    # # 5-fold
    # data_dict_index = np.arange(len(train_val_data_dict['data']))
    # cv = KFold(n_splits=5, shuffle=True)
    # for train_data_idx, val_data_idx in cv.split(data_dict_index):
    #     train_data_dict = {k: train_val_data_dict[k][train_data_idx] for k in train_val_data_dict}
    #     val_data_dict = {k: train_val_data_dict[k][val_data_idx] for k in train_val_data_dict}
    #     # print('train_data_idx', train_data_idx)
    #     # print('val_data_idx', val_data_idx)
    #
    #     train_ds = BCIIV2ADataset(train_data_dict)
    #     val_ds = BCIIV2ADataset(val_data_dict)
    #
    #     data, label, _ = train_ds[0]
    #     print(data.shape)
    #     print(label)
    #     break

    # test label

    # path = glob.glob(os.path.join(root_2a, 'true_labels/*E.mat'))
    # print(path)
    # for p in path:
    #     data = scio.loadmat(p)['classlabel']
    #     print(data)
    #     print(data.shape)
    #     break
    filter_order = 5
    fs = 250
    data_len = 3
    window_size = 1
    stride = 0.5
    wavelet = 'cgau1'
    cwt_scale = 20
    # train_data_dict = read_gdf_data(root_2a,
    #                                 '*1T.gdf',
    #                                 # '*E.gdf',
    #                                 filter_order=filter_order,
    #                                 data_len=data_len,
    #                                 window_size=window_size,
    #                                 stride=stride,
    #                                 f_start=0.5,
    #                                 f_end=45)
    # test_data_dict = read_gdf_data(root_2a,
    #                                '*1E.gdf',
    #                                # '*E.gdf',
    #                                filter_order=filter_order,
    #                                data_len=data_len,
    #                                window_size=window_size,
    #                                stride=stride,
    #                                f_start=0.5,
    #                                f_end=45,
    #                                mat_label='true_labels/')
    # print(test_data_dict['label'])
    # print(test_data_dict['label'].shape)
    # train_ds = BCIIV2ADataset(train_data_dict)
    # test_ds = BCIIV2ADataset(test_data_dict)
    # print('train mean', train_ds.mean.T, '\n train std', train_ds.std.T)
    # print('test mean', test_ds.mean.T, '\n test std', test_ds.std.T)

    # check min max values
    check_scale()
