import cv2
import numpy as np

from utils.pose import PoseEstimator
from utils.sort import SORT


def main():
    estimator = PoseEstimator(dst_size=(960, 540))
    tracker = SORT()

    video = "./samples/walking_man.mp4"
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        exists_next, frame = cap.read()
        if not exists_next:
            break
        frame = cv2.resize(frame, (1280, 768))  # experimental trial

        # get predictions of pose estimator
        preds = estimator.estimate(frame)  # type: list
        preds = np.array(preds)  # shape: (n, 56)

        bboxes = preds[:, :5]  # shape: (n, 5)  # each row: (x1, y1, x2, y2, conf)
        kptss = preds[:, 5:].reshape(-1, 17, 3)  # 17 keypoints * 3 columns (x, y, conf)
