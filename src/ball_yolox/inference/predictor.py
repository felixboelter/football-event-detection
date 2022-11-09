from ball_yolox.utils import postprocess, vis
from ball_yolox.data import ValTransform
import os
import cv2
import numpy as np
import torch
from loguru import logger
import time
from itertools import accumulate

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=None,
        trt_file=None,
        decoder=None,
        device="cpu"
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=None)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
#             with torch.autocast('cuda'):
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic = True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info
    
    def batch_inference(self, imgs):
        
        height, width = imgs[0].shape[:2]

        ratio = min(self.test_size[0] / imgs[0].shape[0], self.test_size[1] / imgs[0].shape[1])

        preproc_imgs = []
        for img in imgs:
            img, _ = self.preproc(img, None, self.test_size)
            preproc_imgs.append(img)
        imgs = torch.from_numpy(np.array(preproc_imgs))
        preproc_imgs = []
        imgs = imgs.float()
        if self.device == "gpu":
#             imgs = imgs.cuda()
            imgs = imgs.cuda().half()

        with torch.no_grad():
            t0 = time.time()
#             with torch.autocast('cuda'):
            outputs = self.model(imgs)
#             logger.info("Infer time 1: {:.4f}s".format(time.time() - t2))
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())

            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic = True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, ratio

    def visual(self, output, img_info, cls_conf=0.0):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

