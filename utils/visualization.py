import cv2
import numpy as np


COLOR_MAP = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'orange': (255, 128, 0),
    'azure': (0, 128, 255)
}

KEYPOINT_MAP = {
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
    16: 'R_ankle'
}

KEYPOINT_CONNECTION_MAP = {
    0: {1: COLOR_MAP['green'], 2: COLOR_MAP['green']},
    1: {3: COLOR_MAP['green']},
    2: {4: COLOR_MAP['green']},
    3: {},
    4: {},
    5: {6: COLOR_MAP['magenta'], 7: COLOR_MAP['red'], 11: COLOR_MAP['magenta']},
    6: {8: COLOR_MAP['orange'], 12: COLOR_MAP['magenta']},
    7: {9: COLOR_MAP['red']},
    8: {10: COLOR_MAP['orange']},
    9: {},
    10: {},
    11: {12: COLOR_MAP['magenta'], 13: COLOR_MAP['blue']},
    12: {14: COLOR_MAP['azure']},
    13: {15: COLOR_MAP['blue']},
    14: {16: COLOR_MAP['azure']},
    15: {},
    16: {}}


def visualize_pose(mat, preds):
    mat_size = mat.shape[:2][::-1]  # (w, h)
    radius = max(int(min(mat_size) * 0.005), 1)
    thickness = max(int(radius * 0.5), 1)

    def draw_bbox(pt1, pt2):
        color = COLOR_MAP["green"]
        cv2.rectangle(mat, pt1, pt2, color, thickness)
    
    def draw_edge(kpts):
        for i in range(len(kpts)):
            adjacency = KEYPOINT_CONNECTION_MAP[i]
            for j in adjacency.keys():
                color = adjacency[j]
                cv2.line(mat, kpts[i], kpts[j], color, thickness)

    def draw_node(kpts):
        color = COLOR_MAP["white"]
        for kpt in kpts:
            cv2.circle(mat, kpt, radius, color, -1)

    for pred in preds:
        x_min, y_min, x_max, y_max = np.array(pred[:4], dtype=int)
        bbox_conf = pred[4]

        kpts = np.array(pred[5:]).reshape(-1, 3)
        kpts_conf = kpts[:, 2]
        kpts = kpts[:, :2].astype(int)

        draw_bbox((x_min, y_min), (x_max, y_max))
        draw_edge(kpts)
        draw_node(kpts)
