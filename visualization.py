import cv2
import numpy as np


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


def visualize_pose(img:np.ndarray, estims, radius, thickness):
    """ Visualize poses and position of persons in the image.
    Args:
        img(np.ndarray): image matrix
        estims(dict): estimations of PoseEstimator
    """
    def draw_connections(kpts):
        for i in range(len(kpts)):
            connection = KEYPOINT_CONNECTION[i]
            for j in connection.keys():
                connection_color = connection[j]
                cv2.line(img, kpts[i], kpts[j], connection_color, thickness)

    def draw_keypoints(kpts):
        color = COLOR['white']
        for kpt in kpts:
            cv2.circle(img, kpt, radius, color, thickness)

    for i, estim in enumerate(estims):
        kpts = estim['kpts']
        kpts = [kpt[:2] for kpt in kpts]
        draw_connections(kpts)
        draw_keypoints(kpts)

        text_scale = 1
        text_thickness = 2
        text_linetype = cv2.LINE_AA

        bbox = estim['bbox']
        cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), thickness)
        cv2.putText(
            img,
            f"person: {str(i)}",
            (bbox[0][0]+20, bbox[0][1]+40),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 255, 0),
            text_thickness,
            text_linetype,
            bottomLeftOrigin=False)