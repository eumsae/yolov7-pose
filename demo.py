import os
import time

import cv2
import numpy as np

from utils.pose import PoseEstimator
from utils.sort import SortTracker
from utils.visualization import draw_tracked_pose


class PoseTracker():
    def __init__(self, video):
        if not os.path.exists(video):
            raise FileNotFoundError(video)
        self.video = video
        self.pose_estimator = PoseEstimator()  # default
        self.sort_tracker = SortTracker()  # default

    def display_prediction(self, input_size=(960, 540)):
        cap = cv2.VideoCapture(self.video)

        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        font_org = (int(height * 0.02), int(height * 0.04))
        font_face = cv2.FONT_HERSHEY_PLAIN
        font_scale = height * 0.0015
        font_color = (0, 255, 0)
        font_thickness = 2

        prev_time = 0

        while cap.isOpened():
            exists, frame = cap.read()
            if not exists:
                break

            preds = np.array(self.pose_estimator.estimate(frame, input_size))
            if preds.ndim == 2:
                boxes = preds[:, :5]
                poses = preds[:, 5:].reshape(-1, 17, 3)

                detection_map = self.sort_tracker.update(boxes)
                for detection_i in detection_map.keys():
                    trk_id = detection_map[detection_i]
                    bbox = boxes[detection_i]
                    pose = poses[detection_i]
                    draw_tracked_pose(frame, trk_id, bbox, pose)

            curr_time = time.time()
            real_fps = round(1 / (curr_time - prev_time), 2)
            prev_time = curr_time

            cv2.putText(
                frame,
                "FPS "+str(real_fps),
                font_org,
                font_face,
                font_scale,
                font_color,
                font_thickness)

            cv2.imshow("result", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    video = "/workspace/pytorch/yolov7-pose/samples/abnormal4.mp4"
    pose_tracker = PoseTracker(video)
    pose_tracker.display_prediction()
