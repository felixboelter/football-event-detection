from ball_yolox.exp import get_exp
from ball_yolox.ball_interpolation import bt_smooth_tracking, get_cropped_frames
from ball_yolox.inference import Predictor
from ball_yolox.utils import get_model_info
from config import YOLOX_Config, SlowFast_Config
from itertools import accumulate
from loguru import logger
import torch, glob, os, cv2, time, sys, shutil, numpy as np, pandas as pd
from tqdm import tqdm
from importlib import import_module
import shutil
from glob import glob
import pytorchvideo
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
import torch.nn as nn
from slowfast.data_loader.data_generator import DataLoader
from slowfast.utils.training_utils import ModelCheckpoint
from slowfast.utils.data_utils import *

class SubmissionGenerator:
    def __init__(self, config, video_paths):
        self.config = config
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        # Model
        print(f' Model: {self.config.architecture} '.center(self.terminal_width, '*'), end='\n\n')
        model_type = import_module('models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config)

        self.model = create_model(self.config)
        # print(self.model)
        self.config.fold = None
        model_checkpoint = ModelCheckpoint(config=self.config, weight_dir='../')
        self.model, _, _, _ = model_checkpoint.load(self.model, load_best=True)

        print(f' Loading data '.center(self.terminal_width, '*'))

        self.video_paths = video_paths

    def _generate_submission_file_model(self, model, outfile_prefix=''):
        print(f' Predict '.center(self.terminal_width, '*'))
        model.eval()
        start_int = (self.config.window_len - self.config.prediction_len) // 2

        vid_preds = {}
        for vid in self.video_paths:
            print(vid)

            data_loader = DataLoader(self.config)
            test_loader = data_loader.create_test_loader(vid)

            pred_range = test_loader.dataset.pred_range
            preds = np.zeros((int(test_loader.dataset.duration * 25), 4))
            sta = (self.config.window_len - self.config.prediction_len) // 2
            norm_factor = self.config.prediction_len // self.config.pred_jump

            for i, [x, idx] in enumerate(tqdm(test_loader)):
                x = [inp.to(self.config.device) for inp in x]
                idx = idx.data.numpy()

                with torch.autocast('cuda'):
                    pred = nn.functional.softmax(model(x), 1).data.cpu().numpy()
                    pred = pred[:, :, start_int:start_int + self.config.prediction_len]

                for b_pred_idx, b_pred in zip(idx, pred):
                    prev = preds[sta:sta+b_pred.shape[1]]
                    preds[sta:sta+b_pred.shape[1]] = prev + (b_pred.T[:prev.shape[0]] / norm_factor)
                    sta += self.config.pred_jump
        
            vid_preds[vid[vid.rfind('/') + 1:-4]] = preds
            
        return vid_preds

        # predictions = np.concatenate(preds, 0).transpose(2, 0, 1).reshape(-1, 28)
        # sample_submission = pd.read_csv('../data/sample_submission_uncertainty.csv')
        #
        # # Merge with sample_submission by using series ids
        # pred_ids = np.concatenate([[series_id[series_id.find('_') + 1:] + '_' + str(q).ljust(5, '0')
        #                             for series_id in self.agg_ids] for q in self.quantiles])
        # predictions_df = pd.DataFrame(np.hstack([pred_ids.reshape(-1, 1), predictions]),
        #                               columns=['id'] + [f'F{i + 1}' for i in range(28)])
        # sample_submission = sample_submission[['id']].merge(predictions_df, how='left',
        #                                                     left_on=sample_submission.id.str[:-11], right_on='id')
        # sample_submission['id'] = sample_submission['id_x']
        # del sample_submission['id_x'], sample_submission['id_y']
        #
        # # Export
        # sample_submission.to_csv(f'{self.sub_dir}/{outfile_prefix}submission.csv.gz', compression='gzip', index=False,
        #                          float_format='%.3g')

    def generate_submission_file(self):
        return self._generate_submission_file_model(self.model)

def get_ball(predictor : Predictor, config : YOLOX_Config, video_path : str, start_time : float = 0.0):
    """
    Given a video, run ball tracking on it and save the cropped frames to a new video.
    
    :param predictor: the YOLOX_Predictor object
    :type predictor: Predictor
    :param config: the configuration object that contains all the parameters for the model
    :type config: YOLOX_Config
    :param video_path: the path to the video you want to track
    :type video_path: str
    :param start_time: the time in seconds to start the video at
    :type start_time: float
    """
    logger.info(f"Starting ball tracking for video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    border = 8
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_no = 0
    save_path = os.path.join(f'./tracking_outputs/{video_path[video_path.rfind("/")+1:-4]}.mp4')
    logger.info(f"Saving video to: {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (config.crop_frame_size[0], config.crop_frame_size[1])
    )
    frames, centers, save_centers = [], [], []
    batch_frames = []
    batch_num = 0
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp >= start_time:
                resized = cv2.resize(frame, (1280, 720))
                frame = cv2.copyMakeBorder(resized, border, border, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
                frame_shape = frame.shape
                batch_frames.append(frame)
                if len(batch_frames) == config.bt_batch_size:
                    outputs, ratio = predictor.batch_inference(batch_frames)
                    frames += batch_frames
                    batch_num += 1
                    if (batch_num % 20) == 0: logger.info(batch_num, timestamp)
                    for output in outputs:
                        if output is not None: 
                            output = output.cpu().numpy()
                            center = ((output[0, 2] / ratio + output[0, 0] / ratio) // 2,
                                      (output[0, 1] / ratio + output[0, 3] / ratio) // 2)
                        else:
                            center = None
                        centers.append(center)
                        batch_frames = []
                # if enough frames have been accumulated, smooth ball tracking and get cropped frames
                if len(frames) > 300:
                    bt_centers = bt_smooth_tracking(centers, n_pts=config.bt_smooth_n_pts, outlier_threshold = config.outlier_thresh)
                    bt_centers = list(accumulate(bt_centers, lambda x, y: y or x))
                    bt_centers = [c if c is not None else (frame.shape[1] // 2, frame.shape[0] // 2)
                                  for c in bt_centers]
                    subset_cropped_frames = get_cropped_frames(frames, bt_centers, config)
                    for _, save_frame in enumerate(subset_cropped_frames):
                        vid_writer.write(save_frame)
                    save_centers += centers
                    frames, centers = [], []
                frame_no += 1
        else:
            # run last batch
            outputs, ratio = predictor.batch_inference(batch_frames)
            frames += batch_frames
            for output in outputs:
                if output is not None: 
                    output = output.cpu().numpy()
                    center = ((output[0, 2] / ratio + output[0, 0] / ratio) // 2, 
                              (output[0, 1] / ratio + output[0, 3] / ratio) // 2)
                else: center = None
                centers.append(center)
            # get rest of the frames
            bt_centers = bt_smooth_tracking(centers, n_pts=config.bt_smooth_n_pts, outlier_threshold = config.outlier_thresh)
            bt_centers = list(accumulate(bt_centers, lambda x, y: y or x))
            bt_centers = [c if c is not None else (frame_shape[1] // 2, frame_shape[0] // 2)
                          for c in bt_centers]
            subset_cropped_frames = get_cropped_frames(frames, bt_centers, config)
            save_centers += centers
            for save_frame in subset_cropped_frames:
                vid_writer.write(save_frame)
            break
                
    cap.release()
    vid_writer.release()

def run_ball_track(exp, config : YOLOX_Config, resized_path : str):
    """
    Loads the model, and then uses the model to predict the ball's location in each frame.
    
    :param exp: the experiment object
    :param config: the configuration file for the YOLO model
    :type config: YOLOX_Config
    :param resized_path: the path to the resized images
    :type resized_path: str
    """
    exp.test_conf = config.conf
    exp.nmsthre = config.nms
    
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    if config.device == "gpu":
        model.cuda().half()
    model.eval()

    ckpt_file = config.yolo_model_ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    predictor = Predictor(
        model, exp, ("ball",), None, None,
        config.device
    )
    get_ball(predictor, config, resized_path)



if __name__ == '__main__':
    YOLOX_config = YOLOX_Config()
    SlowFast_config = SlowFast_Config()
    try: os.mkdir('./tracking_outputs')
    except: pass
    for vid in glob.glob('./data/clips/*'):
        exp = get_exp(YOLOX_config.exp_file, None)
        run_ball_track(exp, YOLOX_config, vid)
    torch.cuda.empty_cache()
    vid_paths = glob.glob('./tracking_outputs/*')
    predictor = SubmissionGenerator(SlowFast_config, vid_paths)
    vid_preds = predictor.generate_submission_file()
