import cv2
import numpy as np

def is_inside_partial(box1, box2, overlap_threshold=0.3, return_ratio=False):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    if xB <= xA or yB <= yA:
        return 0 if return_ratio else False  # no intersection
    
    intersection_area = (xB - xA) * (yB - yA)
    hand_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    overlap_ratio = intersection_area / float(hand_area)

    if return_ratio:
        return overlap_ratio
    return overlap_ratio >= overlap_threshold


def boxes_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)
