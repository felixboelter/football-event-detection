import sys

import numpy as np

sys.path.extend(['..'])

import torch.utils.data
import torch.utils.data as data_utils
import os
import torchvision
import pytorchvideo.transforms.functional as tranfs
import pytorchvideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths

from slowfast.utils.data_utils import *


# Dataset (Input Pipeline)
class CustomDataset(data_utils.Dataset):
    """
    Custom dataset

    Arguments:

    Returns:
    """

    def __init__(self, config=None, eval_mode=False, validation=False, training=True, video_path=None):

        self.cfg = config
        self.eval_mode = eval_mode
        self.validation = validation
        self.video_path = video_path
        self.training = training
        self.testing = False if self.video_path is None else True

        if self.testing:
            self.test_encoded_vid = EncodedVideo.from_path(video_path)
            self.duration = float(self.test_encoded_vid.duration)
            self.pred_range = np.arange(0, self.duration - (self.cfg.window_len * 0.04),
                                        self.cfg.pred_jump * 0.04)

        batch_label_sizes = [int(self.cfg.batch_size * i) for i in self.cfg.label_ratios]
        batch_label_sizes[-1] += sum(batch_label_sizes) - self.cfg.batch_size
        self.batch_label_sizes = batch_label_sizes

        self.tv_transforms = torch.nn.Sequential(
            torchvision.transforms.Grayscale()
        )

        if not self.eval_mode:
            data_df = pd.read_csv(self.cfg.label_csv)

            train_data = data_df.loc[(data_df['video_id'].isin(self.cfg.train_vids))
                                     & (data_df['event'].isin(self.cfg.labels))]
            val_data = data_df.loc[data_df['video_id'].isin(self.cfg.val_vids) & (data_df['event'].isin(self.cfg.labels))]

            del train_data['event_attributes']
            del val_data['event_attributes']

            train_data['event'] = train_data['event'].map(self.cfg.label_dict)
            val_data['event'] = val_data['event'].map(self.cfg.label_dict)

            # transform data to have prediction corresponding to frames
            train_pr_data = train_data.copy()
            train_pr_data = train_pr_data.reset_index(drop=True)
            train_pr_data['time'] = round(train_pr_data['time'] * 25) / 25
            val_pr_data = val_data.copy()
            val_pr_data = val_pr_data.reset_index(drop=True)
            val_pr_data['time'] = round(val_pr_data['time'] * 25) / 25

            if validation:
                self.val_data_raw = val_data
                self.val_pr_data = get_labels_all_frames(val_pr_data)
                self.val_data_by_label = [np.array(self.val_pr_data.loc[self.val_pr_data['event'] == i].index)
                                     for i in range(len(self.cfg.labels) + 1)]

                val_encoded_vids = {}
                for vid in self.cfg.val_vids:
                    val_encoded_vids[vid] = pytorchvideo.data.encoded_video.EncodedVideo.from_path(
                        os.path.join(self.cfg.vid_paths, vid + '_tracking_cropped.mp4'))
                self.val_encoded_vids = val_encoded_vids
                print('Loading and encoding of videos completed')

                # # keep a fixed dataset for validation testing
                # self.val_data_X, self.val_data_y = [], []
                self.val_data_size = int(self.cfg.epoch_steps * self.cfg.batch_size * 0.25)
                # for s in range(self.val_data_size):
                #     batch_data = get_random_batch(val_pr_data, val_data_by_label, self.val_encoded_vids, self.cfg)
                #     self.val_data_X.append(np.array(batch_data[0]))
                #     self.val_data_y.append(np.array(batch_data[1]))
                #
                # self.val_data_X = np.concatenate(self.val_data_X)
                # self.val_data_y = np.concatenate(self.val_data_y)

            if training:
                self.train_pr_data = get_labels_all_frames(train_pr_data)
                self.tr_data_by_label = [np.array(self.train_pr_data.loc[self.train_pr_data['event'] == i].index)
                                         for i in range(len(self.cfg.labels) + 1)]

                # load and encode videos
                tr_encoded_vids = {}
                for vid in self.cfg.train_vids:
                    tr_encoded_vids[vid] = pytorchvideo.data.encoded_video.EncodedVideo.from_path(
                        os.path.join(self.cfg.vid_paths, vid + '_tracking_cropped.mp4'))

                print('Loading and encoding of videos completed')
                self.tr_encoded_vids = tr_encoded_vids

    def __len__(self):
        if self.validation:
            return self.val_data_size
        elif self.training:
            return self.cfg.epoch_steps * self.cfg.batch_size
        else:
            return len(self.pred_range)

    def __getitem__(self, idx):
        if self.testing:
            int_s = self.pred_range[idx]
            int_e = int_s + (self.cfg.window_len * 0.04)

            clip = self.test_encoded_vid.get_clip(int_s, int_e)['video']
            clip = preprocess_clip(clip, self.cfg)

            return [clip, idx]

        if self.validation:
            pr_data = self.val_pr_data
            data_by_label = self.val_data_by_label
            encoded_vids = self.val_encoded_vids
        else:
            pr_data = self.train_pr_data
            data_by_label = self.tr_data_by_label
            encoded_vids = self.tr_encoded_vids

        # duplicate code from utils function, but oh well... don't have time
        # randomly select label according to ratios
        lab = np.random.choice(range(len(self.cfg.labels) + 1), p=self.cfg.label_ratios)

        # randomly select events according to label
        label_idx = np.random.choice(data_by_label[lab])
        label_times = pr_data.iloc[label_idx]['time']
        tol_window = self.cfg.tolerances[lab]

        if lab != 3:  # not background label
            # randomly select start of the interval
            start_pred_int = np.random.choice(range(tol_window + 1, self.cfg.prediction_len - tol_window))
        else:
            start_pred_int = self.cfg.prediction_len - 1

        # get interval start and end timestamps (with respect to the event)
        pred_win_start = (self.cfg.window_len - self.cfg.prediction_len) // 2
        int_idx = np.array([-pred_win_start - start_pred_int,
                            (self.cfg.window_len - pred_win_start - self.cfg.prediction_len) +
                            (self.cfg.prediction_len - start_pred_int)])
        int_times = np.array([i * 0.04 for i in int_idx]) + label_times

        # get video clips and labels
        item_idx, item_t = (int_idx + label_idx).T, int_times.T
        vid_id = pr_data.iloc[item_idx[0]]['video_id']
        lab_data = np.array(pr_data.iloc[item_idx[0]: item_idx[1]]['event'])[
                   pred_win_start : pred_win_start + self.cfg.prediction_len]
        lab_data[start_pred_int - tol_window:start_pred_int + tol_window] = lab

        if lab_data.shape[0] < self.cfg.prediction_len:
            append = np.array([3] * (self.cfg.prediction_len - lab_data.shape[0]))
            if int_times[0] < 1500:  # append at start
                lab_data = np.concatenate([append, lab_data])
            else:
                lab_data = np.concatenate([lab_data, append])
        clip = encoded_vids[vid_id].get_clip(item_t[0], item_t[1])['video']
        if clip.shape[1] < self.cfg.window_len:
            append = torch.zeros(clip.shape[0], self.cfg.window_len - clip.shape[1], clip.shape[2], clip.shape[3])
            if int_times[0] < 1500:  # append at start
                clip = torch.concat((append, clip), dim=1)
            else:
                clip = torch.concat((clip, append), dim=1)

        if self.cfg.aug and (not self.validation):
            if np.random.random() < self.cfg.aug_scale_p:
                clip = tranfs.random_resized_crop(clip, clip.shape[2], clip.shape[3],
                                                  scale=(self.cfg.aug_scale[0], self.cfg.aug_scale[1]),
                                                  aspect_ratio=(1, 1))
            if np.random.random() < self.cfg.aug_hflip_p:
                clip = torchvision.transforms.functional.hflip(clip)

        # # convert to grayscale
        # clip = self.tv_transforms(clip.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        clip = preprocess_clip(clip, self.cfg)

        return [clip, torch.from_numpy(lab_data).long()]


class DataLoader:
    def __init__(self, config):
        self.cfg = config

    def create_train_loader(self):
        dataset = CustomDataset(self.cfg, False, False, True)

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                           num_workers=3)

    def create_val_loader(self):
        dataset = CustomDataset(self.cfg, False, True, False)

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                           num_workers=3)

    def create_test_loader(self, video_path):
        self.dataset = CustomDataset(self.cfg, True, False, False, video_path)

        return torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.cfg.batch_size, shuffle=False,
                                           num_workers=0)
