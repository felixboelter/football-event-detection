import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import torch
from torchvision.transforms._functional_video import normalize
from pytorchvideo.transforms.functional import uniform_temporal_subsample, short_side_scale


def get_labels_all_frames(data_df):
    # add labels for each frame
    start_times = np.floor(data_df.groupby('video_id')['time'].min())
    end_times = np.ceil(data_df.groupby('video_id')['time'].max())

    full_data = []
    for vid_id in start_times.index:
        sta = start_times.loc[start_times.index == vid_id].values[0]
        end = end_times.loc[end_times.index == vid_id].values[0]

        vid_df = pd.DataFrame(np.arange(sta, end, 0.04), columns=['time'])

        # merge and get labels
        vid_df['time'] = round(vid_df['time'] * 100).astype(int)
        lab_df = data_df.loc[data_df['video_id'] == vid_id][['time', 'event']].copy()
        lab_df['time'] = round(lab_df['time'] * 100).astype(int)

        vid_df = pd.merge(vid_df, lab_df, how='left', on=['time'])
        vid_df['time'] = vid_df['time'] / 100
        vid_df = vid_df.fillna(3)
        vid_df['event'] = vid_df['event'].astype(int)

        vid_df['video_id'] = vid_id

        full_data.append(vid_df)

    full_data = pd.concat(full_data)
    full_data = full_data.reset_index(drop=True)
    full_data['event_grp'] = (full_data['event'].groupby(full_data['event'].ne(
        full_data['event'].shift()).cumsum()).cumsum() / full_data['event']).fillna(0).astype(int)

    return full_data


def get_random_batch(full_data, data_by_label, encoded_vids, cfg):
    batch_label_sizes = [int(cfg.batch_size * i) for i in cfg.label_ratios]
    batch_label_sizes[-1] += sum(batch_label_sizes) - cfg.batch_size

    # get random clip interval according to required label
    batch_X, batch_y = [], []
    for lab in range(len(cfg.labels) + 1):
        # randomly select events according to label
        label_idx = np.random.choice(data_by_label[lab], batch_label_sizes[lab])
        label_times = full_data.iloc[label_idx]['time'].values

        if lab != 3:  # not background label
            # randomly select start of the interval
            start_pred_int = np.random.choice(cfg.prediction_len, batch_label_sizes[lab])
        else:
            start_pred_int = np.array([cfg.prediction_len - 1] * batch_label_sizes[lab])

        # get interval start and end timestamps (with respect to the event)
        pred_win_start = (cfg.window_len - cfg.prediction_len) // 2
        int_idx = np.array([-pred_win_start - start_pred_int,
                            (cfg.window_len - pred_win_start - cfg.prediction_len) +
                            (cfg.prediction_len - start_pred_int)])
        int_times = np.array([i * 0.04 for i in int_idx]) + label_times

        # get video clips and labels
        for item_idx, item_t in zip((np.array(int_idx) + label_idx).T, int_times.T):
            vid_id = full_data.iloc[item_idx[0]]['video_id']
            lab_data = np.array(full_data.iloc[item_idx[0]: item_idx[1]]['event'])[pred_win_start:
                                                                                   pred_win_start + cfg.prediction_len]
            clip = encoded_vids[vid_id].get_clip(item_t[0], item_t[1])['video']

            batch_X.append(clip)
            batch_y.append(lab_data)

    return batch_X, batch_y


def preprocess_clip(clip, cfg):
    # Image [0, 255] -> [0, 1].
    clip = short_side_scale(clip, cfg.img_size)
    clip = uniform_temporal_subsample(clip, cfg.window_len)
    clip = clip.float() / 255.0

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(cfg.data_mean, dtype=np.float32),
        np.array(cfg.data_std, dtype=np.float32),
    )

    # generate both pathways
    fast_pathway = clip
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        clip,
        1,
        torch.linspace(
            0, clip.shape[1] - 1, clip.shape[1] // cfg.slow_fast_alpha
        ).long(),
    )
    clip = [slow_pathway, fast_pathway]

    return clip

