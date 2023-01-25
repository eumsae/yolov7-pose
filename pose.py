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


if not os.path.exists(W6PT_PATH):
    import wget
    pose_url = 'https://github.com/WongKinYiu/yolov7/' \
                + 'releases/download/v0.1/yolov7-w6-pose.pt'
    print('Download yolov7-w6-pose.pt ...')
    wget.download(pose_url, YOLO_PATH)
    print('\nDone.')


COLOR = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'orange': (255, 128, 0),
    'azure': (0, 128, 255)}

KEYPOINT = {
    0: 'nose',
    1: 'L_eye',
    2: 'R_eye',
    3: 'L_ear',
    4: 'R_ear',
    5: 'L_shoulder',
    6: 'R_shoulder',
    7: 'L_elbow',
    8: 'R_elbow',
    9: 'L_wrist',
    10: 'R_wrist',
    11: 'L_hip',
    12: 'R_hip',
    13: 'L_knee',
    14: 'R_knee',
    15: 'L_ankle',
    16: 'R_ankle'}

KEYPOINT_CONNECTION = {
    0: {1: COLOR['green'], 2: COLOR['green']},
    1: {3: COLOR['green']},
    2: {4: COLOR['green']},
    3: {},
    4: {},
    5: {6: COLOR['magenta'], 7: COLOR['red'], 11: COLOR['magenta']},
    6: {8: COLOR['orange'], 12: COLOR['magenta']},
    7: {9: COLOR['red']},
    8: {10: COLOR['orange']},
    9: {},
    10: {},
    11: {12: COLOR['magenta'], 13: COLOR['blue']},
    12: {14: COLOR['azure']},
    13: {15: COLOR['blue']},
    14: {16: COLOR['azure']},
    15: {},
    16: {}}


def visualize_pose(mat, estims, radius, thickness):
    """ Visualize poses and position of persons in the image.
    Args:
        mat(np.ndarray): image matrix
        estims(dict): estimations of PoseEstimator
    """
    def draw_connections(kpts):
        for i in range(len(kpts)):
            connection = KEYPOINT_CONNECTION[i]
            for j in connection.keys():
                connection_color = connection[j]
                cv2.line(mat, kpts[i], kpts[j], connection_color, thickness)

    def draw_keypoints(kpts):
        color = COLOR['white']
        for kpt in kpts:
            cv2.circle(mat, kpt, radius, color, thickness)

    for estim in estims:
        kpts = estim['kpts']
        kpts = [kpt[:2] for kpt in kpts]
        draw_connections(kpts)
        draw_keypoints(kpts)


class PoseEstimator():
    def __init__(self):
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch_device = torch.device(self.torch_device)

        weights = torch.load(W6PT_PATH, self.torch_device)
        self.torch_model = weights['model']
        self.torch_model.float().eval()
        if torch.cuda.is_available():
            self.torch_model.half().to(self.torch_device)

        self.input_size = (960, 960)
        self.stride = 64
        self.conf_thres = 0.25
        self.iou_thres = 0.65
        self.resize_info = None

    def estimate(self, mat):
        mat = self._resize_img(mat)
        tsr = self._transform_img2tensor(mat)
        preds = self._predict(tsr)
        estims = self._postproc(preds)
        return estims

    def estimate_with_visualization(self, mat, radius, thickness):
        estims = self.estimate(mat)
        visualize_pose(mat, estims, radius, thickness)
        return estims

    def _resize_img(self, mat):
        resized_mat, self.resize_info = \
            resize_img_keeping_aspect_ratio(
                mat, self.input_size, self.stride)
        return resized_mat

    def _transform_img2tensor(self, mat):
        tsr = transforms.ToTensor()(mat)
        tsr = torch.tensor(np.array([tsr.numpy()]))
        if torch.cuda.is_available():
            tsr = tsr.half().to(self.torch_device)
        return tsr

    def _predict(self, tsr):
        with torch.no_grad():
            preds, _ = self.torch_model(tsr)
        preds = non_max_suppression_kpt(
            prediction=preds,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            nc=self.torch_model.yaml['nc'],
            kpt_label=True)
        preds = output_to_keypoint(preds)
        return preds

    def _postproc(self, preds):
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
        for pred in preds:
            cx, cy, w, h = pred[2:6].tolist()  # bounding box
            p1, p2 = cxcywh2pp(cx, cy, w, h)
            bbox = restore(p1), restore(p2)

            kpts = pred[7:].T.reshape(-1, 3).tolist()  # keypoints
            for i in range(len(kpts)):
                kpts[i][0], kpts[i][1] = restore((kpts[i][0], kpts[i][1]))

            estim = {'bbox': bbox, 'kpts': kpts}
            estims.append(estim)
        return estims


if __name__ == '__main__':
    mat = cv2.imread('sample.png')
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    estimator = PoseEstimator()
    _ = estimator.estimate_with_visualization(mat, 1, 2)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    cv2.imwrite('sample_out.png', mat)