import torch
class YOLOX_Config:
    def __init__(self):
        self.video_path = './9f4df856_0.mp4'
        self.yolo_model_ckpt = './models/ball_tracking_model.pth'
        self.exp_file = './models/yolox_exp.py'
        self.nms = 0.5
        self.conf = 0.6
        self.device = "cpu"
        
        # ball tracking
        self.outlier_thresh = 20
        self.bt_smooth_n_pts = 2
        self.bt_batch_size = 64

        #cropping
        self.crop_frame_size = [256, 256]

class SlowFast_Config:
    def __init__(self):
        self.resume_training = False

        self.loss_fn = 'CrossEntropyLoss'
        self.metric = 'AccMetric'
        self.architecture = 'slowfast'

        self.num_epochs = 200
        self.epoch_steps = 1600
        self.batch_size = 16
        self.grad_accum = 1
        self.learning_rate = 0.001
        self.window_len = 100  # number of frames in each window sent to model
        self.prediction_len = 24  # number of frames for which the model will predict events
        self.label_ratios = [0.28, 0.55, 0.12, 0.05]
        self.tolerances = [3, 5, 3, 0]
        self.img_size = 128
        self.size_fact = 256 // self.img_size
        self.pred_jump = 24

        self.data_mean = [0.45, 0.45, 0.45]
        self.data_std = [0.225, 0.225, 0.225]
        self.slow_fast_alpha = 4

        # augmentation
        self.aug = True
        self.aug_hflip_p = 0.5
        self.aug_scale = [0.7, 0.9]
        self.aug_scale_p = 0.2

        self.vid_paths = '/kaggle/working/tracking_outputs/'
        self.label_csv = 'train.csv'

        self.train_vids = ['3c993bd2_0', '3c993bd2_1', '1606b0e6_0', '1606b0e6_1',
                           'cfbe2e94_0', 'cfbe2e94_1', '35bd9041_0', '35bd9041_1', '4ffd5986_0']
        self.val_vids = ['407c5a9e_1', 'ecf251d4_0', '9a97dae4_1']
        self.labels = ['challenge', 'play', 'throwin']
        self.label_dict = {v: i for i, v in enumerate(self.labels)}


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')