import cv2
import sys
import os
from os.path import join
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.core.render.mesh_renderer.glutils.meshutil import perspective
from gibson2.core.render.profiler import Profiler
from gibson2.utils.assets_utils import get_model_path
import math
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import utils


def blcam2lookat(blcam):
    x, y, z, rx, ry, rz = blcam
    eye = np.array([x, y, z])
    # initial up and viewing vector
    up = np.array([0, 1, 0])
    view = np.array([0, 0, -1])
    # rotate up and viewing vector by euler angles
    view = utils.rot_x(rx) @ view
    view = utils.rot_y(ry) @ view
    view = utils.rot_z(rz) @ view
    up = utils.rot_x(rx) @ up
    up = utils.rot_y(ry) @ up
    up = utils.rot_z(rz) @ up

    target = eye + view
    return eye, target, up


def init_mesh_renderer(mesh_path, intrinsic, height=480, width=640, cuda_idx=0):
    renderer = MeshRenderer(width=width, height=height, device_idx=cuda_idx)
    renderer.load_object(mesh_path)

    renderer.add_instance(0)
    # fov_y
    cx, cy, fx, fy = intrinsic[0, 2], intrinsic[1, 2], intrinsic[0, 0], intrinsic[1, 1]
    assert fx == fy and cx == width/2 and cy == height/2, "igibson mesh renderer assume fx == fy and cx == width/2 and cy == height/2"
    renderer.set_fov(np.rad2deg(2 * math.atan(height / (2 * fy))))
    return renderer

def load_mesh(renderer, mesh_path):
    idx = len(renderer.instances)
    renderer.load_object(mesh_path)
    renderer.add_instance(idx)

def mesh_render(renderer, blcam, depth_clip=True, hidden=()):
    eye, target, up = blcam2lookat(blcam)
    renderer.set_camera(eye, target, up)

    assert len(renderer.instances) <= 2         # full mesh & seq mesh
    # Hide sequence mesh if not specified
    if not hidden and len(renderer.instances) == 2:
        hidden = (renderer.instances[1], )

    color, depth = renderer.render(modes=('rgb', '3d'), hidden=hidden)
    color = cv2.cvtColor((color.clip(0, 1)[..., :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    depth = (-depth[..., 2] * 1000).astype(np.uint16)
    if depth_clip:
        depth[depth < 0.2e3] = 0
        depth[depth > 5e3] = 0
    return color, depth

def gl_render_from_blcam(work_dir, seq_name, intrinsic):
    mesh_path = join(work_dir, 'mesh/scene_mesh_lowres.obj')
    pose_path = join(work_dir, 'seq', f'blcam_{seq_name}.txt')
    img_list_path = join(work_dir, 'seq', f'img_{seq_name}.txt')
    save_dir = join(work_dir, 'gl_images', seq_name)
    os.makedirs(save_dir, exist_ok=True)
    renderer = init_mesh_renderer(mesh_path, intrinsic)
    print(renderer.get_intrinsics())

    with open(pose_path, 'r') as f:
        with open(img_list_path, 'w') as tr:
            for idx, blcam in enumerate(f):
                blcam = list(blcam.strip().split(' '))
                blcam = [float(i) for i in blcam]
                color, depth = mesh_render(renderer, blcam)

                cv2.imwrite(join(save_dir, f'{idx:06d}_color.png'), color)
                cv2.imwrite(join(save_dir, f'{idx:06d}_depth.png'), depth)
                np.savetxt(join(save_dir, f'{idx:06d}_pose.txt'), utils.blcam2pose(blcam))
                print(join(save_dir, f'{idx:06d}_depth.png'))
                tr.write(join(save_dir, f'{idx:06d}_depth.png') + '\n')

def gl_render_from_blcam_and_output(work_dir, output_dir, seq_name, intrinsic):
    mesh_path = join(work_dir, 'mesh/scene_mesh_lowres.obj')
    pose_path = join(work_dir, 'seq', f'blcam_{seq_name}.txt')
    print(pose_path)
    img_list_path = join(work_dir, 'seq', f'img_{seq_name}.txt')
    output_img_list_path = join(work_dir, 'seq', f'img_{seq_name}.txt')
    save_dir = join(output_dir, 'gl_images', f'{seq_name}')
    os.makedirs(save_dir, exist_ok=True)
    renderer = init_mesh_renderer(mesh_path, intrinsic)
    print(renderer.get_intrinsics())

    out_cams = []

    with open(pose_path, 'r') as f:
        with open(img_list_path, 'w') as tr:
            for idx, blcam in enumerate(f):
                blcam = list(blcam.strip().split(' '))
                blcam = [float(i) for i in blcam]
                out_cams.append(blcam)
                color, depth = mesh_render(renderer, blcam)

                cv2.imwrite(join(save_dir, f'{idx:06d}_color.png'), color)
                cv2.imwrite(join(save_dir, f'{idx:06d}_depth.png'), depth)
                np.savetxt(join(save_dir, f'{idx:06d}_pose.txt'), utils.blcam2pose(blcam))
                print(join(save_dir, f'{idx:06d}_depth.png'))
                tr.write(join(save_dir, f'{idx:06d}_depth.png') + '\n')


