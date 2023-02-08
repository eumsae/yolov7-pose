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
    def __init__(self, stride=64, conf_thres=0.25, iou_thres=0.65):
        #self.dst_size = dst_size
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
    
    def estimate(self, mat):
        mat, info = resize_keeping_aspect_ratio(mat, mat.shape[:2][::-1], self.stride)
        #mat, info = resize_keeping_aspect_ratio(mat, self.dst_size, self.stride)
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

    dst_size = fit(width), fit(height)

    return dst_size


def calc_resize_factors(src_size, dst_size):
    w1, h1 = src_size
    w2, h2 = dst_size

    rw = w2 / w1  # ratio
    rh = h2 / h1

    resize_factors = rw, rh

    return resize_factors


def calc_gap(src_size, dst_size):
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
    border_type = cv2.BORDER_CONSTANT
    top, bottom, left, right = thickness

    bordered_mat = cv2.copyMakeBorder(
        mat,
        top, bottom, left, right,
        border_type, value=rgb,)

    return bordered_mat


def resize(mat, resize_factors):
    src_size = np.array(mat.shape[:2][::-1])
    dst_size = np.round(src_size * resize_factors).astype(int)

    resized_mat = cv2.resize(mat, dst_size)

    return resized_mat


def resize_keeping_aspect_ratio(mat, dst_size, stride=None):
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

    return resized_mat, (resize_factors, diff_origin)


if __name__ == "__main__":
    from utils.visualization import visualize_pose

    conf_thres = 0.25
    estimator = PoseEstimator(conf_thres=conf_thres)

    video = "./samples/pedestrians.mp4"
    assert os.path.exists(video), "Not exists the video.mp4"

    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = resize_keeping_aspect_ratio(frame, (960, 540), 64)
        visualize_pose(frame, estimator.estimate(frame))
        cv2.imshow("res", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
