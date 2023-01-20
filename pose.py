# Author: Seunghyun Kim

import os
import sys
from pathlib import Path


YOLO_PATH = str(Path(__file__).absolute().parents[0])
W6PT_PATH = os.path.join(YOLO_PATH, 'yolov7-w6-pose.pt')
sys.path.append(YOLO_PATH)


import cv2
import numpy as np
import torch
from torchvision import transforms

from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
from images import resize_img_keeping_aspect_ratio
from visualization import visualize_pose


if not os.path.exists(W6PT_PATH):
    import wget
    pose_url = 'https://github.com/WongKinYiu/yolov7/' \
                + 'releases/download/v0.1/yolov7-w6-pose.pt'
    print('Download yolov7-w6-pose.pt ...')
    wget.download(pose_url, YOLO_PATH)
    print('\nDone.')


class PoseEstimator():
    def __init__(self, conf_thres=0.25, iou_thres=0.65):
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch_device = torch.device(self.torch_device)

        weights = torch.load(W6PT_PATH, self.torch_device)
        self.torch_model = weights['model']
        self.torch_model.float().eval()
        if torch.cuda.is_available():
            self.torch_model.half().to(self.torch_device)

        self.input_size = (960, 960)
        self.stride = 64
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.resize_info = None

    def estimate_pose(self, img:np.ndarray):
        img = self._resize(img)
        img_tensor = self._convert_to_tensor(img)
        estims = self._estimate(img_tensor)
        estims = self._postproc(estims)
        return estims

    def estimate_pose_with_visualization(
        self,
        img:np.ndarray,
        keypoint_radius:int,
        limb_thickness:int
    ):
        estims = self.estimate_pose(img)
        visualize_pose(img, estims, keypoint_radius, limb_thickness)
        return estims

    def _resize(self, img:np.ndarray):
        resized_mat, self.resize_info = \
            resize_img_keeping_aspect_ratio(
                img, self.input_size, self.stride)
        return resized_mat

    def _convert_to_tensor(self, img:np.ndarray):
        tensor = transforms.ToTensor()(img)
        tensor = torch.tensor(np.array([tensor.numpy()]))
        if torch.cuda.is_available():
            tensor = tensor.half().to(self.torch_device)
        return tensor

    def _estimate(self, tensor):
        with torch.no_grad():
            estims, _ = self.torch_model(tensor)
        estims = non_max_suppression_kpt(
            prediction=estims,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            nc=self.torch_model.yaml['nc'],
            kpt_label=True)
        estims = output_to_keypoint(estims)
        return estims

    def _postproc(self, estims):
        resize_factors = self.resize_info['resize_factors']
        diff_origin = self.resize_info['diff_origin']

        def restore(p):
            x, y = p  # point
            x = int(round((x - diff_origin[0]) / resize_factors[0]))
            y = int(round((y - diff_origin[1]) / resize_factors[1]))
            return x, y

        def cxcywh2pp(cx, cy, w, h):
            p1 = int(round(cx - w / 2)), int(round(cy - h / 2))
            p2 = int(round(cx + w / 2)), int(round(cy + h / 2))
            return p1, p2

        estims = []
        for pred in estims:
            # bounding-box
            cx, cy, w, h = pred[2:6].tolist()
            p1, p2 = cxcywh2pp(cx, cy, w, h)
            bbox = restore(p1), restore(p2)
            # keypoints
            kpts = pred[7:].T.reshape(-1, 3).tolist()
            for i in range(len(kpts)):
                kpts[i][0], kpts[i][1] = restore((kpts[i][0], kpts[i][1]))

            estim = {'bbox': bbox, 'kpts': kpts}
            estims.append(estim)
        return estims


if __name__ == '__main__':
    img = cv2.imread('sample.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    estimator = PoseEstimator()
    _ = estimator.estimate_pose_with_visualization(img, 1, 2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('sample_out.png', img)