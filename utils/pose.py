import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

YOLO_PATH = Path(__file__).absolute().parents[1]
W6PT_PATH = YOLO_PATH / "yolov7-w6-pose.pt"
sys.path.append(YOLO_PATH)

from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

if not os.path.exists(W6PT_PATH):
    import wget
    w6_pose_url = "https://github.com/WongKinYiu/yolov7/" \
                + "releases/download/v0.1/yolov7-w6-pose.pt"
    print("Download yolov7-w6-pose.pt ...")
    wget.download(w6_pose_url, str(YOLO_PATH))
    print("\nDone.")


class PoseEstimator():
    def __init__(self, stride=64, conf_thres=0.25, iou_thres=0.65):
        self.stride = stride
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.resize_factors = (1.0, 1.0)
        self.diff_origin = (0, 0)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        weights = torch.load(W6PT_PATH, map_location=self.device)
        self.model = weights["model"]
        self.model.float().eval()
        if torch.cuda.is_available():
            self.model.half().to(self.device)

    def estimate(self, mat, input_size=(960, 540)):
        mat, info = resize_keeping_aspect_ratio(mat, input_size, self.stride)
        self.resize_factors, self.diff_origin = info

        tsr = transforms.ToTensor()(mat)
        tsr = torch.tensor(np.array([tsr.numpy()]))
        if torch.cuda.is_available():
            tsr = tsr.half().to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.model(tsr)
        outputs = non_max_suppression_kpt(
            prediction=outputs,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            nc=self.model.yaml["nc"],  # num classes
            kpt_label=True)
        outputs = output_to_keypoint(outputs)
        
        preds = []
        for output in outputs:
            # bounding box
            x, y, w, h, conf = output[2:7]
            x_min, y_min = self._restore_coord(x - w / 2, y - h / 2)
            x_max, y_max = self._restore_coord(x + w / 2, y + h / 2)
            bbox = [x_min, y_min, x_max, y_max, conf]  # len: 5

            # keypoints
            kpts = output[7:].reshape(-1, 3)
            for i, (x, y) in enumerate(kpts[:, :2]):
                kpts[i, :2] = self._restore_coord(x, y)
            kpts = kpts.reshape(-1).tolist()  # len: 51 (17 keypoints * 3 columns(x, y, conf))

            pred = bbox + kpts  # total len: 56
            preds.append(pred)
        return preds
    
    
    def _restore_coord(self, x, y):
        x = (x - self.diff_origin[0]) / self.resize_factors[0]
        y = (y - self.diff_origin[1]) / self.resize_factors[1]
        return x, y


def fit_size_to_stride(src_size, stride):
    def fit(x):
        if x % stride:
            x = x - (x % stride) + stride
        return x
    
    width, height = src_size

    input_size = fit(width), fit(height)

    return input_size


def calc_resize_factors(src_size, input_size):
    w1, h1 = src_size
    w2, h2 = input_size

    rw = w2 / w1  # ratio
    rh = h2 / h1

    resize_factors = rw, rh

    return resize_factors


def calc_gap(src_size, input_size):
    w1, h1 = src_size
    w2, h2 = input_size
    if w1 > w2 or h1 > h2:
        msg = "Width and height of the dst size must be " \
            + "greater than that of the src size."
        raise ValueError(msg)
    
    half_w = (w2 - w1) / 2
    half_h = (h2 - h1) / 2

    top = int(round(half_h - 0.1))
    bottom = int(round(half_h + 0.1))
    left = int(round(half_w - 0.1))
    right = int(round(half_w + 0.1))

    gap = top, bottom, left, right

    return gap


def add_border(mat, thickness, rgb=(0,0,0)):
    border_type = cv2.BORDER_CONSTANT
    top, bottom, left, right = thickness

    bordered_mat = cv2.copyMakeBorder(
        mat,
        top, bottom, left, right,
        border_type, value=rgb,)

    return bordered_mat


def resize(mat, resize_factors):
    src_size = np.array(mat.shape[:2][::-1])
    input_size = np.round(src_size * resize_factors).astype(int)

    resized_mat = cv2.resize(mat, input_size)

    return resized_mat


def resize_keeping_aspect_ratio(mat, input_size, stride=None):
    if stride is not None:
        input_size = fit_size_to_stride(input_size, stride)

    src_size = mat.shape[:2][::-1]  # (width, height)
    resize_factors = calc_resize_factors(src_size, input_size)
    resize_factors = (min(resize_factors),) * 2  # (min, min)
    resized_mat = resize(mat, resize_factors)
    
    src_size = resized_mat.shape[:2][::-1]  # (width, height)
    gap = calc_gap(src_size, input_size)
    diff_origin = gap[2], gap[0]  # (top, left)
    resized_mat = add_border(resized_mat, gap)

    return resized_mat, (resize_factors, diff_origin)
