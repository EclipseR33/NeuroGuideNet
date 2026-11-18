from utils import exponential_moving_standardize

import torch
from torch.utils.data import Dataset
import numpy as np
import mne
import os
from scipy.io import loadmat
from copy import deepcopy
import resampy


class OpenBMIDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

        self.ch_names = [
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

    def __len__(self):
        return len(self.data_dict['data'])

    def __getitem__(self, idx):
        subject_id = int(self.data_dict['subject_id'][idx])
        run_idx = self.data_dict['run_idx'][idx]
        data = self.data_dict['data'][idx]
        label = self.data_dict['label'][idx]

        data = torch.as_tensor(data, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.long)
        subject_id = torch.as_tensor(subject_id, dtype=torch.long)
        return data, label, subject_id


def extract_events(t, y_dec, fs, filtered_data, data_len, run_idx_count):
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

    data_dict = {'cue_idx': [], 'run_idx': [], 'data': [], 'label': []}
    cue_cnt = 0
    for i, (start_idx, label) in enumerate(zip(t.tolist(), y_dec.tolist())):
        start_idx = round(start_idx)
        end_idx = round(start_idx + data_len * fs)

        data = filtered_data[:, start_idx:end_idx]

        if len(data.shape) == 2:
            data = data[:, :, None]
        assert len(data.shape) == 3

        # plt.figure()
        # plt.plot(data[:10, :, 0].T)
        # plt.title("Sample channels")
        # plt.show()

        # plt.figure()
        # plt.contourf(np.arange(data[:,:,0].shape[0]),
        #              np.arange(data[:,:,0].shape[1]),
        #              abs(data[:,:,0].T))
        # plt.subplots_adjust(hspace=0.4)
        # plt.show()

        data_dict['cue_idx'].append(cue_cnt)
        data_dict['run_idx'].append(run_idx_count)
        data_dict['data'].append(data)
        data_dict['label'].append(label)

        cue_cnt += 1
    return data_dict


def resample_cnt(cnt, new_fs, old_t):
    if new_fs == cnt.info['sfreq']:
        return deepcopy(cnt)

    data = cnt.get_data().T

    new_data = resampy.resample(
        data, cnt.info["sfreq"], new_fs, axis=0, filter="kaiser_fast"
    ).T
    old_fs = cnt.info["sfreq"]
    new_info = mne.create_info(cnt.info['ch_names'], new_fs)
    t = old_t * new_fs / float(old_fs)
    return mne.io.RawArray(new_data, new_info), t


def read_mat_data(root, subject_select, session, filter_order,
                  data_len, fs, f_start, f_end,
                  factor_new=0.001, is_train=True):

    all_data_dict = {'subject_id': [], 'run_idx': [], 'data': [], 'label': []}
    for sess in session:
        for subject in subject_select:
            print(f'sess: {sess} | subject: {subject}')
            path = os.path.join(root, 'sess{:02d}_subj{:02d}_EEG_MI.mat'.format(int(sess), int(subject)))

            data = loadmat(path)
            if is_train:
                data = data['EEG_MI_train'][0]
            else:
                data = data['EEG_MI_test'][0]

            x = data['x'][0].T
            old_t = data['t'][0][0]
            old_fs = data['fs'][0]
            y_dec = data['y_dec'][0][0] - 1
            classes = data['class'][0]
            ch_names = [i[0] for i in data['chan'][0][0]]

            print(ch_names)
            info = mne.create_info(ch_names=ch_names, sfreq=old_fs, ch_types='eeg')
            raw = mne.io.RawArray(x, info, verbose="WARNING")
            raw.load_data()
            raw.set_eeg_reference(ref_channels='average')
            raw, t = resample_cnt(raw, fs, old_t)
            # raw.plot()

            if not isinstance(f_start, list):
                filtered_data = raw.copy().filter(
                    l_freq=f_start, h_freq=f_end, method='iir',
                    iir_params={'order': filter_order, 'ftype': 'butter'},
                    picks='all'
                ).get_data()

                filtered_data = exponential_moving_standardize(filtered_data,
                                                               factor_new=factor_new,
                                                               init_block_size=int(raw.info['sfreq'] * 4))
            else:
                filtered_data = []
                for low, high in zip(f_start, f_end):
                    temp_data = raw.copy().filter(
                        l_freq=low, h_freq=high, method='iir',
                        iir_params={'order': filter_order, 'ftype': 'butter'},
                        picks='all'
                    ).get_data()

                    temp_data = exponential_moving_standardize(temp_data,
                                                               factor_new=factor_new,
                                                               init_block_size=int(raw.info['sfreq'] * 4))
                    # temp_data = temp_data.astype(np.float32) # for FBCNet OpenBMI cross
                    filtered_data.append(temp_data)
                filtered_data = np.stack(filtered_data, -1)

            subject_data_dict = extract_events(t, y_dec, fs, filtered_data, data_len, session)
            all_data_dict['subject_id'] += [subject for _ in range(len(subject_data_dict['run_idx']))]
            all_data_dict['run_idx'] += subject_data_dict['run_idx']
            all_data_dict['data'] += subject_data_dict['data']
            all_data_dict['label'] += subject_data_dict['label']
    # print(all_data_dict['subject_id'])
    return {k: np.array(all_data_dict[k]) for k in all_data_dict}


if __name__ == '__main__':
    root = r'F:\AI\Dataset\OpenBMI'
    subject_select = [1]
    session = [1]
    train_data_dict = read_mat_data(
        root, subject_select, session,
        filter_order=5,
        data_len=4,
        fs=250,
        f_start=0.5,
        f_end=45,
        factor_new=0.001,
        is_train=True
    )
    print(train_data_dict)
