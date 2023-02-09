import cv2
import numpy as np

from utils.pose import PoseEstimator
from utils.sort import SORT
from visualization import visualize_pose

conf_thres = 0.25
iou_thres = 0.65

estimator = PoseEstimator()
mot_tracker = SORT(max_age=1, min_hits=3, iou_threshold=iou_thres)

video = "./samples/walking_man.mp4"
cap = cv2.VideoCapture(video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (960, 540))
    preds = estimator.estimate(frame)

    dets = np.array(preds)
    if dets.ndim == 2:
        dets = dets[:, :5]

    trackers = mot_tracker.update(dets)
    for d in trackers:
        print(d)

    visualize_pose(frame, preds)
    cv2.imshow("res", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()