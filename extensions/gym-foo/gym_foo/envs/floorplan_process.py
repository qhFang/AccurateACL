import os
import yaml
import cv2
import sys
from os.path import join
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import utils
import imutils


def blcam2pixel(yaml_path, blcam):
    """ ALl in meters """
    bl_x, bl_y = blcam
    with open(yaml_path, 'r') as y:
        data = yaml.safe_load(y)

    origin = data['origin']
    assert origin[0] == origin[1], "floor.png origin[0] != origin[1]"
    ori = origin[0]
    img_coord = (int(round((-bl_y - ori) * 100)), int(round((bl_x - ori) * 100)))
    return img_coord


def crop_center(img, yaml_path, blcam, len_x, len_y=None):
    """ ALl in meters """
    pixel_center = blcam2pixel(yaml_path, blcam[:2])
    if len_y is None:
        len_y = len_x
    len_x *= 100
    len_y *= 100
    view_angle = blcam[-1]  # rad

    # order: bottom left, top left, top right, bottom right
    src = np.array([
        [[len_x/2, -len_y/2]],
        [[-len_x/2, -len_y/2]],
        [[-len_x/2, len_y/2]],
        [[len_x/2, len_y/2]],
    ], np.float32)
    crop = rotated_crop(img, src, view_angle, pixel_center)

    # wo rotation
    # rint = lambda x: int(round(x))
    # crop = img[rint(center_x - len_x / 2): rint(center_x + len_x / 2),
    #        rint(center_y - len_y / 2): rint(center_y + len_y / 2)]
    return crop


def rotated_crop(img, src, angle, pixel_center):
    # (x, y) coordinate
    rot_mat = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    src = np.matmul(rot_mat, src.reshape(4, 2, 1)).reshape(4, 2).astype(np.float32)

    center_x, center_y = pixel_center
    src[:, 0] += center_x
    src[:, 1] += center_y

    # (x, y) coordinate -> (u, v) coordinate
    src = src[:, ::-1]

    # cv2.drawContours(img, [src.astype(np.int)], 0, (0, 0, 255), 2)
    rect = cv2.minAreaRect(src.reshape(4, 1, 2).astype(np.int))
    width = int(rect[1][0])
    height = int(rect[1][1])
    dst = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype=np.float32)

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped


def draw_training_seq(img_path, yaml_path, blcams):
    color = (128, 128, 128)
    img = cv2.imread(img_path)
    for blcam in blcams:
        coord = blcam2pixel(yaml_path, (blcam[0], blcam[1]))
        cv2.circle(img, coord[::-1], 2, color, thickness=10)
        cam_len = 3
        end_point = (blcam[0] - cam_len * np.sin(blcam[-1]), blcam[1] + cam_len * np.cos(blcam[-1]))
        coord_end_point = blcam2pixel(yaml_path, end_point)
        # cv2.circle(img, coord_end_point[::-1], 2, (0, 0, 255), thickness=3)
        cv2.line(img, coord[::-1], coord_end_point[::-1], color, thickness=1)
    return img


def seq2points(blcams, save_path, color=None):
    with open(save_path, 'w') as f:
        for blcam in blcams:
            blcam[:3] *= 1000
            if not color:
                f.write(f'{blcam[0]} {blcam[1]} {blcam[2]}\n')
            else:
                f.write(f'{blcam[0]} {blcam[1]} {blcam[2]} {color[0]} {color[1]} {color[2]}\n')


