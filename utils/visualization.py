import cv2


COLORS = (
    (242, 117, 26),
    (106, 0, 167),
    (143, 13, 163),
    (176, 42, 143),
    (202, 70, 12),
    (224, 100, 97),
    (241, 130, 76),
    (252, 166, 53),
    (252, 204, 37),
    (64, 67, 135),
    (52, 94, 141),
    (41, 120, 142),
    (32, 143, 140),
    (34, 167, 132),
    (66, 190, 113),
    (121, 209, 81),
    (186, 222, 39))


NODES = [
    "nose",
    "L_eye",
    "R_eye",
    "L_ear",
    "R_ear",
    "L_shoulder",
    "R_shoulder",
    "L_elbow",
    "R_elbow",
    "L_wrist",
    "R_wrist",
    "L_hip",
    "R_hip",
    "L_knee",
    "R_knee",
    "L_ankle",
    "R_ankle"]


EDGES = {
    0: (1, 2),
    1: (3,),
    2: (4,),
    3: (),
    4: (),
    5: (6, 7, 11),
    6: (8, 12),
    7: (9,),
    8: (10,),
    9: (),
    10: (),
    11: (12, 13),
    12: (14,),
    13: (15,),
    14: (16,),
    15: (),
    16: ()}


def draw_tracked_pose(mat, trk_id, bbox, pose):
    bbox_xyxy = bbox[:4].astype(int)
    bbox_conf = round(bbox[4], 2)

    kpts_xy = pose[:, :2].astype(int)
    #kpts_conf = pose[:, 2]

    color = COLORS[trk_id % len(COLORS)]
    radius = max(int(min(mat.shape[:2][::-1]) * 0.005), 1)
    thickness = max(int((radius * 0.5)), 1)

    # draw bounding box
    cv2.rectangle(mat, bbox_xyxy[:2], bbox_xyxy[2:], color, thickness)
    cv2.putText(mat, str(trk_id), bbox_xyxy[:2], cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    cv2.putText(mat, str(bbox_conf), (bbox_xyxy[0], bbox_xyxy[3]), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    # draw bones
    for i in range(len(kpts_xy)):
        edges = EDGES[i]
        for j in edges:
            cv2.line(mat, kpts_xy[i], kpts_xy[j], color, thickness)

    # draw joints
    for i, xy in enumerate(kpts_xy):
        cv2.circle(mat, xy, radius, color, -1)
