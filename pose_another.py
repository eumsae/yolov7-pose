import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

YOLO_PATH = Path(__file__).absolute().parents[0]
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
    def __init__(self, dst_size=(960, 960), stride=64, conf_thres=0.25, iou_thres=0.65):
        self.dst_size = dst_size
        self.stride = stride
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        weights = torch.load(W6PT_PATH, map_location=self.device)
        self.model = weights["model"]
        self.model.float().eval()
        if torch.cuda.is_available():
            self.model.half().to(self.device)
    
    def estimate(self, mat):
        """ Estimate pose.
        Args:
            mat(np.ndarray): image matrix
        Returns:
            ...
        """
        mat, _ = resize_keeping_aspect_ratio(mat, self.dst_size, self.stride)
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
        
        preds = []
        for output in outputs:
            xywh = output[2:6].tolist()  # bounding box, len:4
            kpts = output[7:].tolist()  # keypoints, len: 51(17*3)

            pred = xywh + kpts  # len: 55(4+51)
            preds.append(pred)
        
        return preds


def fit_size_to_stride(src_size, stride):
    """ Fit the src size to multiple of the stride.
    Args:
        src_size(list-like, [int, int]): width, height
        stride(int):
    Returns:
        dst_size(tuple, (int, int)): width, height
    """
    def fit(x):
        if x % stride:
            x = x - (x % stride) + stride
        return x
    
    width, height = src_size

    dst_size = fit(width), fit(height)

    return dst_size


def calc_resize_factors(src_size, dst_size):
    """ Calculate resize factors.
    Args:
        src_size(list-like, [int, int]): width, height
        dst_size(list-like, [int, int]): width, height
    Returns:
        resize_factors(tuple, (float, float)):
    """
    w1, h1 = src_size
    w2, h2 = dst_size

    rw = w2 / w1  # ratio
    rh = h2 / h1

    resize_factors = rw, rh

    return resize_factors


def calc_gap(src_size, dst_size):
    """ Calculate the gap.
    Args:
        src_size(list-like, [int, int]): width, height
        dst_size(list-like, [int, int]): width, height
    Returns:
        gap(tuple, (int, int, int, int)): top, bottom, left, right
    Raise:
        ValueError
    """
    w1, h1 = src_size
    w2, h2 = dst_size
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
    """ Add border to the image.
    Args:
        mat(np.ndarray): image matrix
        thickness(list-like, [int, int, int, int]): top, bottom, left, right
        rgb(list-like, [int, int, int]):
    Returns:
        bordered_mat(np.ndarray): bordered image matrix
    """
    border_type = cv2.BORDER_CONSTANT
    top, bottom, left, right = thickness

    bordered_mat = cv2.copyMakeBorder(
        mat,
        top, bottom, left, right,
        border_type, value=rgb,)

    return bordered_mat


def resize(mat, resize_factors):
    """ Resize the image with factors.
    Args:
        mat(np.ndarray): image matrix
        resize_factors(list-like, [float, float]):
    Returns:
        resized_mat(np.ndarray):
    """
    src_size = np.array(mat.shape[:2][::-1])
    dst_size = np.round(src_size * resize_factors).astype(int)

    resized_mat = cv2.resize(mat, dst_size)

    return resized_mat


def resize_keeping_aspect_ratio(mat, dst_size, stride=None):
    """ Resize the image to the dst size with keeping aspect ratio.
    Args:
        mat(np.ndarray): image matrix
        dst_size(list-like, [int, int]): width, height
        stride(int):
    Returns:
        resized_mat(np.ndarray): resized image matrix
        resize_info(dict):
    """
    if stride is not None:
        dst_size = fit_size_to_stride(dst_size, stride)

    src_size = mat.shape[:2][::-1]  # (width, height)
    resize_factors = calc_resize_factors(src_size, dst_size)
    resize_factors = (min(resize_factors),) * 2  # (min, min)
    resized_mat = resize(mat, resize_factors)
    
    src_size = resized_mat.shape[:2][::-1]  # (width, height)
    gap = calc_gap(src_size, dst_size)
    diff_origin = gap[2], gap[0]  # (top, left)
    resized_mat = add_border(resized_mat, gap)
    resize_info = {
        "resize_factors": resize_factors,
        "diff_origin": diff_origin}

    return resized_mat, resize_info