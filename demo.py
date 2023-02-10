import os

import cv2
import numpy as np

from utils.pose import PoseEstimator, EDGES
from utils.sort import SortTracker


class PoseTracker():
    def __init__(self, video):
        if not os.path.exists(video):
            raise FileNotFoundError(video)
        self.video = video
        self.pose_estimator = PoseEstimator()  # default
        self.sort_tracker = SortTracker()  # default

    def display_prediction(self, input_size=(960, 540)):
        cap = cv2.VideoCapture(self.video)
        while cap.isOpened():
            exists, frame = cap.read()  # get frame image
            if not exists:
                break
            preds = np.array(self.pose_estimator.estimate(frame, input_size))
            if preds.ndim == 2:
                boxes = preds[:, :5]
                poses = preds[:, 5:].reshape(-1, 17, 3)

                detection_map = self.sort_tracker.update(boxes)
                for detection_i in detection_map.keys():
                    bbox = boxes[detection_i, :4].astype(int)
                    pose = poses[detection_i, :, :2].astype(int)

                    trk_id = detection_map[detection_i]
                    self.draw_bbox(frame, bbox, trk_id)
                    self.draw_pose(frame, pose, trk_id)

            cv2.imshow("result", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()

    def draw_bbox(self, mat, box, trk_id):
        cv2.putText(mat, str(trk_id), (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        cv2.rectangle(mat, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    def draw_pose(self, mat, pose, trk_id):
        node_radius = 3
        edge_thickness = 2

        for i in range(len(pose)):
            conn = EDGES[i]
            for j in conn.keys():
                color = conn[j]
                cv2.line(mat, pose[i], pose[j], color, edge_thickness)
        for kpt in pose:
            cv2.circle(mat, kpt, node_radius, (0, 0, 255), -1)


if __name__ == "__main__":
    video = "/workspace/pytorch/yolov7-pose/samples/pedestrians.mp4"
    pose_tracker = PoseTracker(video)
    pose_tracker.display_prediction()