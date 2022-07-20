import time
import os
import os.path as osp
from os.path import join
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import numpy as np
import cv2
import yaml
import glob
from natsort import natsorted
from matplotlib import pyplot as plt
import matplotlib
import global_setting as gls

###
# visualize the trajectory record
###


m = matplotlib.cm.ScalarMappable(cmap='Blues')

def blcam2travxy(ori, bl_x, bl_y):
    img_coord = (int(round((-bl_y - ori) * 100)), int(round((bl_x - ori) * 100)))
    return img_coord

def get_color(x):
    r, g, b, a = plt.cm.winter(int(x * 255))
    return r, g, b

def get_color_id(np_array):
    # `to_rgba` returns [r, g, b, a] while we only want [r, g, b]
    color_coding = m.to_rgba(np_array, bytes=True, norm=False)[:, :-1]
    return color_coding

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_errmap(scene_name):
    map_path = osp.join(gls.data_root, scene_name, f'mesh_{args.dynamic}') if args.dynamic  else osp.join(gls.data_root, scene_name, 'mesh')
    map_name = osp.join(map_path, 'floor_render.png')
    errmap = cv2.imread(map_name)
    errmap = cv2.resize(errmap, (1600, 1600))
    assert errmap is not None, f'Image {map_name} does not exist'
    return errmap

def env_render(errmap, info_file):
    errmap = errmap.copy()
    info_lines = info_file.readlines()
    traj_len = len((info_lines))
    traj_color = get_color_id(np.arange(traj_len) / traj_len)
    prev_cam = None
    for i_line, line in enumerate(info_lines):
        line = line.strip("\n").split(" ")
        true_cam = np.array(line[:6]).astype(np.float32)
        stat = int(line[6])

        # agent size
        grid_size = 7

        if stat == 1:
            cam_color = [50, 200, 50]  # success color: green
        elif stat == -1:
            cam_color = [0, 50, 255]  # failure color: red
        else:
            cam_color = [255, 255, 255]  # normal color: white
        # blcam 2 traversable
        x_id, y_id = blcam2travxy(trav_ori, true_cam[0], true_cam[1])

        #errmap[x_id - grid_size: x_id + grid_size,
        #y_id - grid_size: y_id + grid_size] = cam_color

        start_point = (x_id, y_id)

        # 1. middle arrow
        if i_line != traj_len - 1 and i_line != 0 and i_line < 0:
            arrow_len = 2 * grid_size
            end_point = (int(round(start_point[0] - arrow_len * np.cos(true_cam[-1]))),
                         int(round(start_point[1] - arrow_len * np.sin(true_cam[-1]))))
            errmap = cv2.arrowedLine(errmap, start_point[::-1], end_point[::-1], cam_color, thickness=4, tipLength=0.6)
        ###################

        # 2. start and end arrow
        if i_line == traj_len - 1:
            arrow_len = 3 * grid_size
            end_point = (int(round(start_point[0] - arrow_len * np.cos(true_cam[-1]))),
                              int(round(start_point[1] - arrow_len * np.sin(true_cam[-1]))))
            errmap = cv2.arrowedLine(errmap, start_point[::-1], end_point[::-1], cam_color, thickness=6, tipLength=0.8)

        # circle
        t_c = (int(traj_color[i_line, 0]), int(traj_color[i_line, 1]), int(traj_color[i_line, 2]))[::-1]

        # line
        if prev_cam is not None:
            errmap = cv2.line(errmap, prev_point[::-1], start_point[::-1], t_c, thickness=20)

        prev_cam = true_cam
        x_id, y_id = blcam2travxy(trav_ori, prev_cam[0], prev_cam[1])
        prev_point = (x_id, y_id)

    cv2.imwrite(osp.join(image_dir, f'hooo.png'), errmap)

    return errmap


def process_train(errmap, info_file, traj_len):
    errmap = errmap.copy()
    info_lines = info_file.readlines()
    for i_line, line in enumerate(info_lines):
        line = line.strip("\n").split(" ")
        true_cam = np.array(line[:6]).astype(np.float32)

        # agent size
        grid_size = 7

        cam_color = [255, 255, 255]  # normal color: white
        # blcam 2 traversable
        x_id, y_id = blcam2travxy(trav_ori, true_cam[0], true_cam[1])

        start_point = (x_id, y_id)

        arrow_len = 6 * grid_size
        end_point = (int(round(start_point[0] - arrow_len * np.cos(true_cam[-1]))),
                          int(round(start_point[1] - arrow_len * np.sin(true_cam[-1]))))
        errmap = cv2.circle(errmap, start_point[::-1], 1, cam_color, thickness=10)
    return errmap

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', type=str, help='the name of the scene')
    parser.add_argument('--dynamic', type=str, default='', help='dynamic testing scene, e.g. `change3`')
    parser.add_argument('--mode', type=str, default='rl', help='action mode',
                        choices=['rl', 'none', 'turn', 'greedy', 'random', 'train_path', 'turn_train_path'])
    args = parser.parse_args()
    scene_name = args.scene_name
    mode = args.mode
    dynamic = args.dynamic
    data_root = osp.join(gls.codes_root, 'vis', mode, scene_name)

    # path
    result_root = f'{data_root}/{scene_name}/'
    print(result_root)
    info_list = natsorted(glob.glob(osp.join(result_root, '*.txt')))
    image_dir = f'{data_root}/{scene_name}/test_video/'
    os.makedirs(image_dir, exist_ok=True)

    # get floorplan origin
    dyn = '_' + dynamic if dynamic else ''
    with open(osp.join(gls.data_root, scene_name, f'mesh{dyn}', 'floor_trav_0.yaml'), 'r') as y:
        data = yaml.safe_load(y)
    origin = data['origin']
    assert origin[0] == origin[1], "[floorplan yaml] origin[0] != origin[1]"
    trav_ori = origin[0]

    # action
    n = 1
    errmap = get_errmap(scene_name)
    for info_path in info_list:
        print(info_path)
        file_name = info_path.split("/")[-1]
        info_file = open(info_path, "r+")
        errmap = env_render(errmap, info_file)
        info_file.close()
