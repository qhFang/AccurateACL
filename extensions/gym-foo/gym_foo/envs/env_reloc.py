import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import os.path as osp
from os.path import join
import subprocess
import psutil
import time
import cv2
import shutil
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append((os.path.dirname(os.path.abspath(__file__))))
import utils
from os.path import dirname
import igibson_mesh_renderer
import floorplan_process as flp
import datetime
import global_setting as gls
from PIL import Image
from torchvision import transforms as tfs
from random import choices, sample
from collections import deque, Counter
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors  
import torch

class EnvReloc(gym.Env):
    """ A clean environment """
    def __init__(self, name, scene_names, dynamic, seq_names, noise_std, step_size, angle_size, success_reward, color_jitter, n_hyp, wocollide=False, finish_condition=5, worker_id=0, scene_idx=0, cpp_cuda=0, render_cuda=0, fixed_pose=False, use_cuda=False):
        super(EnvReloc, self).__init__()        
        assert scene_idx < len(scene_names), f"scene_idx {scene_idx} out of range of scene_names {scene_names}"
        scene_name = scene_names[scene_idx]
        if '-' in scene_name:
            self.add_num = int(scene_name.split('-')[1])
            scene_name = scene_name.split('-')[0]
        else:
            self.add_num = 0
 
        
        seq_name = seq_names[scene_idx]
        self.scene_name = scene_name
        self.seq_name = seq_name
        #self.seq_name = 'seq-50cm-60deg-half'
        self.scene_dir = join(gls.data_root, scene_name)
        self.mesh_dir = join(self.scene_dir, 'mesh' if not dynamic else f'mesh_{dynamic}')
        self.use_cuda = use_cuda

        self.dist = step_size
        self.rz_dist = np.deg2rad(angle_size)
        print(f'step size: {self.dist}m, angle size: {self.rz_dist / np.pi:.2f} pi')
        self.img_h, self.img_w = 480, 640
        self.intrinsic = np.array([
            [480, 0, 320],
            [0, 480, 240],
            [0, 0, 1],
        ])

        self.time_punish = 0.1
        self.finish = [finish_condition, finish_condition]
        self.n_hyp = n_hyp
        print(f'\n{n_hyp} hypos\n')
        with open(join(self.scene_dir, 'boundary.txt'), 'r') as f:
            self.x_min = float(f.readline().strip())
            self.x_max = float(f.readline().strip())
            self.y_min = float(f.readline().strip())
            self.y_max = float(f.readline().strip())
        self.rz_min, self.rz_max = 0, 2 * np.pi - 1e-4

        self.exclusion = False
        if os.path.exists(join(self.scene_dir, 'exclusion.txt')):
            self.exclusion = True
            print('\nAssign exclusion\n')
            with open(join(self.scene_dir, 'exclusion.txt'), 'r') as ex:
                self.ex_x_min = float(ex.readline().strip())
                self.ex_x_max = float(ex.readline().strip())
                self.ex_y_min = float(ex.readline().strip())
                self.ex_y_max = float(ex.readline().strip())

        self.n_action = 3
        self.action_space = spaces.Discrete(self.n_action)
        self.success_reward = success_reward
        self.color_jitter = color_jitter
        self.print_color_jitter = True if self.color_jitter else False
        self.pre_var = 0

        self.x_bound = [self.x_min - 2, self.x_max + 2]
        self.y_bound = [self.y_min - 2, self.y_max + 2]
        self.z_bound = [0, 2]
        self.r_bound = [0, 2 * np.pi]
        obs_low = np.tile(np.array([self.x_bound[0], self.y_bound[0], self.z_bound[0],
                                    0, 0, 0], np.float32), (self.n_hyp, 1))
        obs_high = np.tile(np.array([self.x_bound[1], self.y_bound[1], self.z_bound[1],
                                     2 * np.pi, 2 * np.pi, 2 * np.pi], np.float32), (self.n_hyp, 1))
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)
        self.traversable_map = cv2.imread(join(self.mesh_dir, 'floor_trav_0_connected.png'), -1)
        with open(join(self.mesh_dir, 'floor_trav_0.yaml'), 'r') as y:
            data = yaml.safe_load(y)
        origin = data['origin']
        assert origin[0] == origin[1], "[floorplan yaml] origin[0] != origin[1]"
        self.ori = origin[0]

        self.wocollide = wocollide

        self.seed()

        self.worker_id = worker_id

        # Online relocalizer
        time_id = datetime.datetime.today().strftime('%m%d_%H%M%S_%f')
        randid = np.random.randint(0, 100000)
        rf_path_name = 'multiple_envs' if '#' in name else name
        self.rf_path = join(self.scene_dir, 'rf_path', rf_path_name, f'{worker_id}_{time_id}_{randid}')
        if len(self.rf_path) > 170:
            self.rf_path = join(self.scene_dir, 'rf_path', 'longpath', f'{worker_id}_{time_id}')
        if osp.exists(self.rf_path):
            print(f'Remove existed rf path {self.rf_path}')
            shutil.rmtree(self.rf_path)
        os.makedirs(self.rf_path, exist_ok=True)
        # self.cpp_model = seq_name.replace('-3dof', '')
        self.cpp_model = seq_name
        model_path = self.scene_dir
        relocalizer_cmd = f'CUDA_VISIBLE_DEVICES={cpp_cuda} ' \
            f'{gls.reloc_root}/build/bin/apps/relocgui/relocgui ' \
            f'-c {self.scene_dir}/intrinsic.txt ' \
            f'--test {join(self.rf_path, "query.txt")} ' \
            f'--model_path {join(model_path, "cpp_model")} ' \
            f'--model {self.cpp_model} ' \
            f'--phase test4rl'
        self.subprocess_reloc = subprocess.Popen(relocalizer_cmd, shell=True)
        print(f'Start relocalizer {self.cpp_model} at pid {self.subprocess_reloc.pid}')
        print(relocalizer_cmd)

        self.fixed_pose = fixed_pose
        if (fixed_pose):
            self.next_idx = -1 + self.add_num
            self.pose_list = np.loadtxt(os.path.join(gls.data_root, f'{scene_name}.txt'))
        # Mesh renderer
        self.mesh_path = join(self.mesh_dir, 'scene_mesh_lowres.obj')
        self.mesh_renderer = igibson_mesh_renderer.init_mesh_renderer(self.mesh_path, self.intrinsic, self.img_h, self.img_w, cuda_idx=render_cuda)

        self.noise_std = noise_std
        if self.noise_std != 0:
            print(f"\nMove step noise injection: std={self.noise_std}\n")

    def reset(self):
        if self.fixed_pose:
            if self.next_idx < 0 + self.add_num:
                self.true_cam = self.pose_list[0]
            else:
                self.true_cam = self.pose_list[self.next_idx]
            self.next_idx += 1
            state = self.relocalize(self.true_cam)
            self.state = state
            return self.state.copy()
        state = np.nan
        collide = True
        exclusion = True
        while np.isnan(state).any() or collide or exclusion:
            x = self.np_random.uniform(self.x_min, self.x_max)
            y = self.np_random.uniform(self.y_min, self.y_max)
            rz = self.np_random.uniform(self.rz_min, self.rz_max)
            self.true_cam = np.array([x, y, 1, np.pi / 2, 0, rz])
            state = self.relocalize(self.true_cam)
            collide = self.collide()
            exclusion = self.init_exclusion()
        self.state = state
        self.hypo = state
        return self.state.copy()

    def reset_assign(self, start_pos):
        self.true_cam = np.asarray(start_pos)
        state = self.relocalize(self.true_cam)
        self.state = state
        self.hypo = state
        return self.state.copy()

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        cam_success = False
        success = False
        done = False
        outbounds = False
        getnan = False
        # get to a new place
        step_noise = np.clip(np.random.normal(0, self.noise_std), -2 * self.noise_std, 2 * self.noise_std)
        rz_noise = np.clip(np.random.normal(0, 2 * self.noise_std), -2 * 2 * self.noise_std, 2 * 2 * self.noise_std)
        self.step_noise = step_noise
        self.rz_noise = rz_noise
        # move forward: x: -sin(rz), y: cos(rz)

        tmp_cam = self.true_cam.copy()        

        if action == 0:
            tmp_cam[0] -= (self.dist + step_noise) * np.sin(tmp_cam[5])
            tmp_cam[1] += (self.dist + step_noise) * np.cos(tmp_cam[5])
        # turn left
        elif action == 1:
            tmp_cam[5] += self.rz_dist + rz_noise
        # turn right
        elif action == 2:
            tmp_cam[5] -= self.rz_dist + rz_noise
        # wrap angles always in range [0, 2pi)
        tmp_cam[5] %= (2 * np.pi)

        # ## results:
        # ## out of bounds / no prediction / succeed / cam_succeed
        # out-of-bounds detection
        if self.collide_cam(tmp_cam):
            reward = -self.success_reward
            if not self.wocollide:
                done = True
            else:
                action = -1
            outbounds = True
        else:
            # get new predictions
            self.true_cam = tmp_cam
            state = self.relocalize(self.true_cam)
            self.state = state
            if np.isnan(state).any():
                # if the relocalizer fails to give predictions, end the episode
                if not self.wocollide:
                    reward = -self.success_reward
                    done = True
                else:
                    reward = -self.time_punish
                getnan = True
            else:
                # Use `state`
                # `state` will always be hypos, `self.state` may be different in different envs
                s = self.cam_succeed(state)
                if s:
                    if not self.wocollide:
                        reward = self.success_reward
                    else:
                        reward = -self.time_punish
                    done = True
                    cam_success = True
                else:
                    reward = -self.time_punish
                if self.succeed(state[0], self.true_cam):
                    success = True


        return self.state.copy(), reward, done, {'succ': success, 'cam_succ': cam_success,
                                                 'outbounds': outbounds, 'getnan': getnan}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pobj = psutil.Process(self.subprocess_reloc.pid)
        for c in pobj.children(recursive=True):
            c.terminate()
        print(f'\nTerminate relocalizer at pid {self.subprocess_reloc.pid}')

        self.mesh_renderer.release()

    def relocalize(self, blcam):
        """
        :param blcam: current blcam
        :return: #hypo * 6 blender-format [X Y Z RX RY RZ] predictions
        """
        # render current view
        self.cur_color, self.cur_depth = igibson_mesh_renderer.mesh_render(self.mesh_renderer, blcam)
        self.cur_color, self.cur_depth = self.cur_color.astype(np.float32), self.cur_depth.astype(np.float32)
        cv2.imwrite(join(self.rf_path, '000000_color.png'), self.cur_color.astype(np.uint8))
        cv2.imwrite(join(self.rf_path, '000000_depth.png'), self.cur_depth.astype(np.uint16))
        # Save an arbitrary pose so that the relocalizer is able to read data
        np.savetxt(join(self.rf_path, '000000_pose.txt'), np.eye(4))

        state = np.zeros((0, 6), np.float32)
        with open(join(self.rf_path, 'query.txt'), 'w') as f:
            f.write(join(self.rf_path, '000000_depth.png'))
        with open(join(self.rf_path, 'ready.txt'), 'w') as f:
            f.write('I am ready')
        # Wait until the cpp relocalizer finishes running and remove the datalist file.
        while os.path.exists(join(self.rf_path, 'ready.txt')):
            pass
        for i in range(self.n_hyp):
            pose = np.loadtxt(join(self.rf_path, f'pose-{i}.txt'), delimiter=',')
            blcam = utils.pose2blcam(pose)
            state = np.vstack((state, blcam[np.newaxis]))
        self.hypo = state
        return state

    def succeed(self, pred_blcam, gt_blcam):
        pred_pose = utils.blcam2pose(pred_blcam)
        gt_pose = utils.blcam2pose(gt_blcam)
        r_err, t_err = utils.rt_err(pred_pose, gt_pose)
        self.err = [r_err, t_err]

        if r_err < self.finish[0] and t_err < self.finish[1]:
            return True
        else:
            return False

    def cam_succeed(self, pred_blcam):
        var = np.var(pred_blcam, axis=0)
        var = var.mean()
        if var < self.var_th:
            return True, var
        else:
            return False, var

    def collide(self):
        trav_xy = utils.blcam2travxy(self.ori, self.true_cam[0], self.true_cam[1])
        trav_value = self.traversable_map[trav_xy] / 255
        assert trav_value == 0 or trav_value == 1
        return False if trav_value else True

    def collide_cam(self, cam):
        trav_xy = utils.blcam2travxy(self.ori, cam[0], cam[1])
        trav_xy = [trav_xy[0], trav_xy[1]]
        if trav_xy[0] < 0:
            trav_xy[0] = 0
        if trav_xy[0] >= self.traversable_map.shape[0]:
            trav_xy[0] = self.traversable_map.shape[0] - 1
        if trav_xy[1] < 0:
            trav_xy[1] = 0
        if trav_xy[1] >= self.traversable_map.shape[1]:
            trav_xy[1] = self.traversable_map.shape[1] - 1
        trav_value = self.traversable_map[trav_xy[0], trav_xy[1]] / 255
        assert trav_value == 0 or trav_value == 1
        return False if trav_value else True

    def check_start_pose(self):
        trav_xy = utils.blcam2travxy(self.ori, self.true_cam[0], self.true_cam[1])
        trav_value = self.special_map[trav_xy] / 255
        assert trav_value == 0 or trav_value == 1
        return False if trav_value else True

    def init_exclusion(self):
        if not self.exclusion:
            return False
        exclusion = self.ex_x_min < self.true_cam[0] < self.ex_x_max and \
            self.ex_y_min < self.true_cam[1] < self.ex_y_max
        return exclusion

    def cam_idx_loc(self, blcam):
        x_id = (blcam[0] - self.x_min) / self.dist
        y_id = (blcam[1] - self.y_min) / self.dist
        rz_id = (blcam[5] - self.rz_min) / self.rz_dist
        return x_id, y_id, rz_id

    def try_action(self, action):
        def temp_outbound():
            if temp_cam[0] < self.x_min or temp_cam[0] >= self.x_max:
                return True
            elif temp_cam[1] < self.y_min or temp_cam[1] >= self.y_max:
                return True
            elif temp_cam[5] < self.rz_min or temp_cam[5] > self.rz_max:
                return True
            else:
                return False

        temp_state = np.zeros((self.n_hyp, 6)) * np.nan
        temp_cam = self.true_cam.copy()

        # get to a new place
        step_noise = np.clip(np.random.normal(0, self.noise_std), -0.1, 0.1)
        # move forward: x -= sin(rz), y += cos(rz)
        if action == 0:
            temp_cam[0] -= (self.dist + step_noise) * np.sin(temp_cam[5])
            temp_cam[1] += (self.dist + step_noise) * np.cos(temp_cam[5])
        # turn left
        elif action == 1:
            temp_cam[5] += self.rz_dist + step_noise
        # turn right
        elif action == 2:
            temp_cam[5] -= self.rz_dist + step_noise
        # wrap angles always in range [0, 2pi)
        temp_cam[5] %= (2 * np.pi)

        # out-bound detection
        if temp_outbound():
            return temp_state
        else:
            # get new predictions
            temp_state = self.relocalize(temp_cam)
            return temp_state

class EnvRelocUncertainty(EnvReloc):
    def __init__(self, name, scene_names, dynamic, seq_names, noise_std, step_size, angle_size, success_reward, color_jitter, n_hyp, wocollide=False, finish_condition=5, worker_id=0, scene_idx=0,  cpp_cuda=0, render_cuda=0, fixed_pose=False, use_cuda=False):
        super(EnvRelocUncertainty, self).__init__(name, scene_names, dynamic, seq_names, noise_std, step_size, angle_size, success_reward, color_jitter, n_hyp, wocollide, finish_condition, worker_id, scene_idx, cpp_cuda, render_cuda, fixed_pose, use_cuda)
        self.state_channel = 73
        self.img_size = 64
        obs_low = -255 * np.ones((self.img_size, self.img_size, self.state_channel), np.float32)
        obs_high = 255 * np.ones((self.img_size, self.img_size, self.state_channel), np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        # load the mesh reconstructed from passive relocalizer training sequence  
        self.hypo_path = join(self.mesh_dir, f'scene_{self.seq_name}_mesh_lowres.obj')
        self.hypo_queue = [mp.Queue(), mp.Queue()]
        igibson_mesh_renderer.load_mesh(self.mesh_renderer, self.hypo_path)

        # load the global point cloud and the uncertainty channel of the world-driven scene map
        filepath = f"{self.scene_dir}/seq/pc_{self.seq_name}_uncertainty_5cm_all.txt"
        global_pc_xyzcon = np.loadtxt(filepath)
        global_pc_xyzcon = global_pc_xyzcon[np.random.choice(np.arange(0, len(global_pc_xyzcon), 1), 2**14)]
        self.global_xyz = global_pc_xyzcon[:, :3].copy()
        self.depth_aware_confidence = global_pc_xyzcon[:, 3:13].copy()
        self.depth_ave_confidence = global_pc_xyzcon[:, 13].copy()

        # build queue for the history channel 
        self.hypo_his = deque(
                              [[0, 0, 0, 0, 0, 0]] * 5,
                              maxlen=5)

        self.coordinate_his = deque([[0] * 2**14] * 5,
                                    maxlen=5)

        self.position_his = deque(
                              [np.zeros((256, 256)).tolist()] * 5,
                              maxlen=5)
        
        # load the uncertainty channel of the camera-driven scene map
        # At first, we need to transfer the map from color to accurate rate by building the colormap
        colormap = get_colormap('jet')
        errmap = cv2.imread(f"{self.scene_dir}/errmap/{self.seq_name}_20cm_errmap_avg.png", -1)
        self.errmap_highres = jet_to_number(colormap, errmap)
        self.errmap = cv2.resize(self.errmap_highres, ((256, 256)))

        # the refresh flag is used to distinguish whether the camera uncertainty has been obtained
        self.refresh = -1

        # start the icp module
        # note that: If we import the icp module from open3d in this file, there will be some conflict between icp module and subprocess module(which is used by rlpyt)
        #            so that we use file connection to instead. And we will find a more efficient way to solve this problem.
        self.icp_path = join(self.rf_path, 'icp')
        if osp.exists(self.icp_path):
            print(f'Remove existed icp path {self.icp_path}')
            shutil.rmtree(self.icp_path)
        os.makedirs(self.icp_path, exist_ok=True)
        icp_cmd = f'python -u {gls.codes_root}/tools/icp.py {self.icp_path} {self.worker_id}'
        print('icp_cmd:', icp_cmd)
        self.icp = subprocess.Popen(icp_cmd, shell=True)
        
        # build the 2d occupancy map to calculate the exploration loss
        self.occupancy_map = np.zeros((self.traversable_map.shape[0] * 2 // 40, self.traversable_map.shape[1] * 2 // 40))

    def reset(self):
        state = super(EnvRelocUncertainty, self).reset()

        # reset the occupancy map
        self.occupancy_map = np.zeros((self.traversable_map.shape[0] * 2 // 40, self.traversable_map.shape[1] * 2 // 40))
        local_position = self.traversable_map.shape
        self.local_position = self.traversable_map.shape
        self.occupancy_map[local_position[0] // 40, local_position[1] // 40] = 1
        local_cam = utils.travxy2blcam(self.ori, local_position[0], local_position[1])
        self.local_cam = np.array([local_cam[0], local_cam[1], 1, np.pi / 2, 0, 0])

        # reset the state
        self.new_state = np.zeros((self.img_size, self.img_size, self.state_channel), np.float32)
        self.get_state(state)
        return self.new_state.copy()

    def reset_assign(self, start_pos):
        state = super(EnvRelocUncertainty, self).reset_assign(start_pos)

        # reset the occupancy map
        self.occupancy_map = np.zeros((self.traversable_map.shape[0] * 2 // 40, self.traversable_map.shape[1] * 2 // 40))
        local_position = self.traversable_map.shape
        self.local_position = self.traversable_map.shape
        self.occupancy_map[local_position[0] // 40, local_position[1] // 40] = 1
        local_cam = utils.travxy2blcam(self.ori, local_position[0], local_position[1])
        self.local_cam = np.array([local_cam[0], local_cam[1], 1, np.pi / 2, 0, 0])

        # reset the state
        self.new_state = np.zeros((self.img_size, self.img_size, self.state_channel), np.float32)
        self.get_state(state)
        return self.new_state.copy()

    def step(self, action):
        state, reward, done, info_dict = super(EnvRelocUncertainty, self).step(action)

        # update the local position and the occupancy map
        if action == 0:
            self.local_cam[0] -= (self.dist + self.step_noise) * np.sin(self.local_cam[5])
            self.local_cam[1] += (self.dist + self.step_noise) * np.cos(self.local_cam[5])
        # turn left
        elif action == 1:
            self.local_cam[5] += self.rz_dist + self.rz_noise
        # turn right
        elif action == 2:
            self.local_cam[5] -= self.rz_dist + self.rz_noise
        # wrap angles always in range [0, 2pi)
        self.local_cam[5] %= (2 * np.pi)
        local_position = utils.blcam2travxy(self.ori, self.local_cam[0], self.local_cam[1])
        self.local_position = local_position

        # update occupancy map
        grid_x, grid_y = local_position[0] // 40, local_position[1] // 40
        if grid_x < 0:
            grid_x = 0
        if grid_y < 0:
            grid_y = 0
        if grid_x >= self.occupancy_map.shape[0]:
            grid_x = self.occupancy_map.shape[0] - 1
        if grid_y >= self.occupancy_map.shape[1]:
            grid_y = self.occupancy_map.shape[1] - 1
        self.occupancy_map[grid_x, grid_y] += 1

        reward += 0.1 / self.occupancy_map[grid_x, grid_y]
        self.get_state(state)

        return self.new_state.copy(), reward, done, info_dict

    def get_state(self, hypos):
        if self.use_cuda:
            self.get_state_cuda(hypos)
            return
        c = self.cur_color.astype(np.uint8)  # uint8 [0, 255]
        if self.color_jitter:
            if self.print_color_jitter:
                print(f"\nColor Jittering\n")
                self.print_color_jitter = False
                # color jittering
                c = Image.fromarray(c)
                if np.random.rand() < 0.5:
                    c = tfs.ColorJitter(brightness=.7)(c)
                if np.random.rand() < 0.5:
                    c = tfs.ColorJitter(contrast=.7)(c)
                if np.random.rand() < 0.5:
                    c = tfs.ColorJitter(saturation=.7)(c)
        if self.refresh == -1:
            if (os.path.exists(join(self.icp_path, 'source_color.png'))):
                os.remove(join(self.icp_path, 'source_color.png'))
                os.remove(join(self.icp_path, 'source_depth.png'))
            cv2.imwrite(join(self.icp_path, 'source_color.png'), self.cur_color.astype(np.uint8))
            cv2.imwrite(join(self.icp_path, 'source_depth.png'), self.cur_depth.astype(np.uint16))
        c = np.array(c).astype(np.float32) / 128 - 1  # (-1, 1)
        c = cv2.resize(c, (self.img_size, self.img_size))
        d = self.cur_depth.astype(np.float32) / 1e3 # depth as meters

        gridy, gridx = np.mgrid[:480, :640]
        x_cam = (gridx - self.intrinsic[0, 2]) / self.intrinsic[0, 0] * d
        y_cam = (gridy - self.intrinsic[1, 2]) / self.intrinsic[1, 1] * d

        x_cam = cv2.resize(x_cam, (self.img_size, self.img_size))
        y_cam = cv2.resize(y_cam, (self.img_size, self.img_size))
        d = cv2.resize(d, (self.img_size, self.img_size))


        self.hypo = hypos

        # If passive relocalizer fails to calculate the camera pose for current frame, use the last estimation.
        if np.isnan(hypos[0]).any():
            hypo = self.hypo_his[-1]
        else:
            hypo = hypos[0]
        pose = utils.blcam2pose(hypo)

        # get the point cloud for current frame
        xyz = np.concatenate((x_cam[..., np.newaxis], y_cam[..., np.newaxis], d[..., np.newaxis]), axis=2)
        xyz = xyz.reshape((xyz.shape[0] * xyz.shape[1], 3))
        tmp_column = np.ones(xyz.shape[0])
        xyz = np.c_[xyz, tmp_column].copy()
        xyz = np.dot(pose, xyz.T).T[:, :3]
        xyz = np.floor(xyz * 10) / 10 

        # get the cloest point from current point cloud to the global point cloud
        current_coordinate = np.zeros((len(self.global_xyz)))
        distance, idx = KDtree_search(xyz, self.global_xyz)

        # delete the outliers and get the current frame channel of world-driven scene map 
        count_idx = Counter(idx[distance <= 0.5])
        unique_idx = np.unique(idx[distance <= 0.5])
        for i in range(len(unique_idx)):
            current_coordinate[unique_idx[i]] = count_idx[unique_idx[i]]

        # get the history channel of world-driven scene map
        self.coordinate_his.append(current_coordinate.copy().tolist())
        coordinate_ave = np.array(self.coordinate_his)
        coordinate_ave = np.average(coordinate_ave, axis=0)


        # If passive relocalizer fails to calculate the camera pose for current frame, use the last estimation.
        if np.isnan(hypos[0]).any():
            current_position_map = np.array(self.position_his[-1])
            distance_transform = np.array(self.position_his[-1])
        else:
            # get the current channel of camera-driven scene map
            trav_xy = utils.blcam2travxy(self.ori, hypos[0][0], hypos[0][1])
            trav_xy = [trav_xy[0], trav_xy[1]]
            current_position_map = self.traversable_map.copy().astype(np.float32)
            if trav_xy[0] < 0:
                trav_xy[0] = 0
            if trav_xy[0] >= current_position_map.shape[0]:
                trav_xy[0] = current_position_map.shape[0] - 1
            if trav_xy[1] < 0:
                trav_xy[1] = 0
            if trav_xy[1] >= current_position_map.shape[1]:
                trav_xy[1] = current_position_map.shape[1] - 1
            # convert the map to gaussian_map
            x = np.linspace(-1, 1, current_position_map.shape[0])
            x = x - x[trav_xy[0]]
            y = np.linspace(-1, 1, current_position_map.shape[1])
            y = y - y[trav_xy[1]]
            x = np.expand_dims(x, -1).repeat(current_position_map.shape[1], -1)
            y = np.expand_dims(y, 0).repeat(current_position_map.shape[0], 0)
            gaussian_map = np.exp(-((pow(x, 2) + pow(y, 2))  / 2 ))
            current_position_map[current_position_map == 0] = -1
            distance_transform = current_position_map.copy()
            current_position_map[current_position_map != -1] = gaussian_map[current_position_map != -1]
            current_position_map = cv2.resize(current_position_map, ((256, 256)))

            # get the distance transform for current channel which will be used to get the history channel
            gridx, gridy = np.mgrid[:distance_transform.shape[0], :distance_transform.shape[1]]
            gridx = gridx - trav_xy[0]
            gridx = gridx * gridx
            gridy = gridy - trav_xy[1]
            gridy = gridy * gridy
            distance_transform[distance_transform != -1] = np.sqrt(gridx + gridy)[distance_transform != -1]

        # get the history channel
        self.position_his.append(cv2.resize(distance_transform.copy(), ((256, 256))))
        position_ave = np.min(np.array(self.position_his), axis=0)

        # get the camera uncertainty
        if np.isnan(hypos[0]).any():
            motion_err = np.array([100, 100])
            self.motion_err = motion_err
        elif self.refresh == -1:
            hypo_c, hypo_d = igibson_mesh_renderer.mesh_render(self.mesh_renderer, hypos[0], hidden = (self.mesh_renderer.instances[0], ))
            if self.refresh == -1:
                if (os.path.exists(join(self.icp_path, 'target_color.png'))):
                    os.remove(join(self.icp_path, 'target_color.png'))
                    os.remove(join(self.icp_path, 'target_depth.png'))
                cv2.imwrite(join(self.icp_path, 'target_color.png'), hypo_c.astype(np.uint8))
                cv2.imwrite(join(self.icp_path, 'target_depth.png'), hypo_d.astype(np.uint16))
                f = open(join(self.icp_path, 'ready.txt'), 'w')
                f.write('ready!')
                f.close()
            while not os.path.exists(join(self.icp_path, 'exclusioned.txt')):
                pass
            while not os.path.exists(join(self.icp_path, 'exclusion.txt')):
                pass

        if not np.isnan(hypos[0]).any(): 
            with open(join(self.icp_path, 'exclusion.txt'), 'r') as f:
                motion_err = f.readlines()
                r_err = float(motion_err[0].rstrip('\n'))
                t_err = float(motion_err[1].rstrip('\n'))
                motion_err = [r_err, t_err]
            self.motion_err = motion_err

            os.remove(join(self.icp_path, 'exclusioned.txt'))
            os.remove(join(self.icp_path, 'exclusion.txt'))
        
        self.refresh = -1

        self.new_state[:, :, :16] = self.errmap.reshape((self.img_size, self.img_size, 16))
        self.new_state[:, :, 16:32] = current_position_map.reshape((self.img_size, self.img_size, 16))
        self.new_state[:, :, 32:48] = position_ave.reshape((self.img_size, self.img_size, 16))
        self.new_state[:, :, 48:60] = self.global_xyz.reshape((self.img_size, self.img_size, 12))
        self.new_state[:, :, 60:64] = self.depth_ave_confidence.reshape((self.img_size, self.img_size, 4))
        self.new_state[:, :, 64:68] = current_coordinate.reshape((self.img_size, self.img_size, 4))
        self.new_state[:, :, 68:72] = coordinate_ave.reshape((self.img_size, self.img_size, 4))
        self.new_state[0, 0, 72] = motion_err[0]
        self.new_state[1, 0, 72] = motion_err[1]

    def get_state_cuda(self, hypos):
        c = self.cur_color.astype(np.uint8)  # uint8 [0, 255]
        if self.color_jitter:
            if self.print_color_jitter:
                print(f"\nColor Jittering\n")
                self.print_color_jitter = False
                # color jittering
                c = Image.fromarray(c)
                if np.random.rand() < 0.5:
                    c = tfs.ColorJitter(brightness=.7)(c)
                if np.random.rand() < 0.5:
                    c = tfs.ColorJitter(contrast=.7)(c)
                if np.random.rand() < 0.5:
                    c = tfs.ColorJitter(saturation=.7)(c)

        if self.refresh == -1:
            if (os.path.exists(join(self.icp_path, 'source_color.png'))):
                os.remove(join(self.icp_path, 'source_color.png'))
                os.remove(join(self.icp_path, 'source_depth.png'))
            cv2.imwrite(join(self.icp_path, 'source_color.png'), self.cur_color.astype(np.uint8))
            cv2.imwrite(join(self.icp_path, 'source_depth.png'), self.cur_depth.astype(np.uint16))

        resize_img_size = Resize([self.img_size, self.img_size])
        resize_256 = Resize([256, 256])

        d = self.cur_depth.astype(np.float32) / 1e3 # depth as meters
        d = torch.from_numpy(d).to('cuda')

        gridy, gridx = torch.meshgrid(torch.arange(480), torch.arange(640))
        gridy = gridy.to('cuda')
        gridx = gridx.to('cuda')
        x_cam = (gridx - self.intrinsic[0, 2]) / self.intrinsic[0, 0] * d
        y_cam = (gridy - self.intrinsic[1, 2]) / self.intrinsic[1, 1] * d

        x_cam = resize_img_size(x_cam.unsqueeze(0)).squeeze(0)
        y_cam = resize_img_size(y_cam.unsqueeze(0)).squeeze(0)
        d = resize_img_size(d.unsqueeze(0)).squeeze(0)

        self.hypo = hypos

        # If passive relocalizer fails to calculate the camera pose for current frame, use the last estimation.
        if np.isnan(hypos[0]).any():
            hypo = self.hypo_his[-1]
        else:
            hypo = hypos[0]
        pose = torch.from_numpy(utils.blcam2pose(hypo).astype(np.float32)).to('cuda')



        # get the point cloud for current frame
        xyz = torch.cat((x_cam.unsqueeze(-1), y_cam.unsqueeze(-1), d.unsqueeze(-1)), axis=2)
        xyz = xyz.reshape((xyz.shape[0] * xyz.shape[1], 3))
        tmp_column = torch.ones(xyz.shape[0]).unsqueeze(-1).to('cuda')
        xyz = torch.cat((xyz, tmp_column), axis=1).clone()
        xyz = torch.mm(pose, xyz.T).T[:, :3]
        xyz = torch.floor(xyz * 10) / 10 

        # get the cloest point from current point cloud to the global point cloud
        current_coordinate = np.zeros((len(self.global_xyz)))
        distance, idx = KDtree_search(xyz.detach().cpu().numpy(), self.global_xyz)

        # delete the outliers and get the current frame channel of world-driven scene map 
        count_idx = Counter(idx[distance <= 0.5])
        unique_idx = np.unique(idx[distance <= 0.5])
        for i in range(len(unique_idx)):
            current_coordinate[unique_idx[i]] = count_idx[unique_idx[i]]

        # get the history channel of world-driven scene map
        self.coordinate_his.append(current_coordinate.copy().tolist())
        coordinate_ave = torch.tensor(self.coordinate_his).to('cuda')
        coordinate_ave = coordinate_ave.mean(axis=0)

        # If passive relocalizer fails to calculate the camera pose for current frame, use the last estimation.
        if np.isnan(hypos[0]).any():
            current_position_map = torch.tensor(self.position_his[-1]).to('cuda')
            distance_transform = torch.tensor(self.position_his[-1]).to('cuda')
        else:
            # get the current channel of camera-driven scene map
            trav_xy = utils.blcam2travxy(self.ori, hypos[0][0], hypos[0][1])
            trav_xy = [trav_xy[0], trav_xy[1]]
            current_position_map = torch.tensor(self.traversable_map.copy().astype(np.float32)).to('cuda')
            if trav_xy[0] < 0:
                trav_xy[0] = 0
            if trav_xy[0] >= current_position_map.shape[0]:
                trav_xy[0] = current_position_map.shape[0] - 1
            if trav_xy[1] < 0:
                trav_xy[1] = 0
            if trav_xy[1] >= current_position_map.shape[1]:
                trav_xy[1] = current_position_map.shape[1] - 1
            # convert the map to gaussian_map
            x = torch.linspace(-1, 1, current_position_map.shape[0]).to('cuda')
            x = x - x[trav_xy[0]]
            y = torch.linspace(-1, 1, current_position_map.shape[1]).to('cuda')
            y = y - y[trav_xy[1]]
            x = x.unsqueeze(-1).repeat(1, current_position_map.shape[1])
            y = y.unsqueeze(0).repeat(current_position_map.shape[0], 1)
            gaussian_map = torch.exp(-((pow(x, 2) + pow(y, 2))  / 2 ))
            current_position_map[current_position_map == 0] = -1
            distance_transform = current_position_map.clone()
            current_position_map[current_position_map != -1] = gaussian_map[current_position_map != -1]
            current_position_map = resize_256(current_position_map.unsqueeze(0)).squeeze(0)

            # get the distance transform for current channel which will be used to get the history channel
            gridx, gridy = torch.meshgrid(torch.arange(distance_transform.shape[0]), torch.arange(distance_transform.shape[1]))
            gridx = gridx.to('cuda')
            gridy = gridy.to('cuda')
            gridx = gridx - trav_xy[0]
            gridx = gridx * gridx
            gridy = gridy - trav_xy[1]
            gridy = gridy * gridy
            distance_transform[distance_transform != -1] = torch.sqrt(gridx + gridy)[distance_transform != -1]

        # get the history channel
        self.position_his.append(resize_256(distance_transform.unsqueeze(0)).squeeze(0).detach().cpu().clone().numpy().tolist())
        position_ave, _ = torch.tensor(self.position_his).min(dim=0)

        # get the camera uncertainty
        if np.isnan(hypos[0]).any():
            motion_err = np.array([100, 100])
            self.motion_err = motion_err
        elif self.refresh == -1:
            hypo_c, hypo_d = igibson_mesh_renderer.mesh_render(self.mesh_renderer, hypos[0], hidden = (self.mesh_renderer.instances[0], ))
            if self.refresh == -1:
                if (os.path.exists(join(self.icp_path, 'target_color.png'))):
                    os.remove(join(self.icp_path, 'target_color.png'))
                    os.remove(join(self.icp_path, 'target_depth.png'))
                cv2.imwrite(join(self.icp_path, 'target_color.png'), hypo_c.astype(np.uint8))
                cv2.imwrite(join(self.icp_path, 'target_depth.png'), hypo_d.astype(np.uint16))
                f = open(join(self.icp_path, 'ready.txt'), 'w')
                f.write('ready!')
                f.close()
            while not os.path.exists(join(self.icp_path, 'exclusioned.txt')):
                pass
            while not os.path.exists(join(self.icp_path, 'exclusion.txt')):
                pass

        if not np.isnan(hypos[0]).any(): 
            with open(join(self.icp_path, 'exclusion.txt'), 'r') as f:
                motion_err = f.readlines()
                r_err = float(motion_err[0].rstrip('\n'))
                t_err = float(motion_err[1].rstrip('\n'))
                motion_err = [r_err, t_err]
            self.motion_err = motion_err

            os.remove(join(self.icp_path, 'exclusioned.txt'))
            os.remove(join(self.icp_path, 'exclusion.txt'))

        self.refresh = -1

        self.new_state[:, :, :16] = self.errmap.reshape((self.img_size, self.img_size, 16))
        self.new_state[:, :, 16:32] = current_position_map.detach().cpu().numpy().reshape((self.img_size, self.img_size, 16))
        self.new_state[:, :, 32:48] = position_ave.detach().cpu().numpy().reshape((self.img_size, self.img_size, 16))
        self.new_state[:, :, 48:60] = self.global_xyz.reshape((self.img_size, self.img_size, 12))
        self.new_state[:, :, 60:64] = self.depth_ave_confidence.reshape((self.img_size, self.img_size, 4))
        self.new_state[:, :, 64:68] = current_coordinate.reshape((self.img_size, self.img_size, 4))
        self.new_state[:, :, 68:72] = coordinate_ave.detach().cpu().numpy().reshape((self.img_size, self.img_size, 4))
        self.new_state[0, 0, 72] = motion_err[0]
        self.new_state[1, 0, 72] = motion_err[1]



    def try_action(self, action):
        def temp_outbound():
            if temp_cam[0] < self.x_min or temp_cam[0] >= self.x_max:
                return True
            elif temp_cam[1] < self.y_min or temp_cam[1] >= self.y_max:
                return True
            elif temp_cam[5] < self.rz_min or temp_cam[5] > self.rz_max:
                return True
            else:
                return False

        temp_state = np.zeros((self.n_hyp, 6)) * np.nan
        temp_cam = self.true_cam.copy()

        # get to a new place
        step_noise = np.clip(np.random.normal(0, self.noise_std), -0.1, 0.1)
        # move forward: x -= sin(rz), y += cos(rz)
        if action == 0:
            temp_cam[0] -= (self.dist + step_noise) * np.sin(temp_cam[5])
            temp_cam[1] += (self.dist + step_noise) * np.cos(temp_cam[5])
        # turn left
        elif action == 1:
            temp_cam[5] += self.rz_dist + step_noise
        # turn right
        elif action == 2:
            temp_cam[5] -= self.rz_dist + step_noise
        # wrap angles always in range [0, 2pi)
        temp_cam[5] %= (2 * np.pi)

        # out-bound detection
        if temp_outbound():
            return 100
        else:
            # get new predictions
            temp_state = self.relocalize(temp_cam)
            if (os.path.exists(join(self.icp_path, 'source_color.png'))):
                os.remove(join(self.icp_path, 'source_color.png'))
                os.remove(join(self.icp_path, 'source_depth.png'))
            cv2.imwrite(join(self.icp_path, 'source_color.png'), self.cur_color.astype(np.uint8))
            cv2.imwrite(join(self.icp_path, 'source_depth.png'), self.cur_depth.astype(np.uint16))
            hypo_c, hypo_d = igibson_mesh_renderer.mesh_render(self.mesh_renderer, temp_state[0], hidden = (self.mesh_renderer.instances[0], ))
            if (os.path.exists(join(self.icp_path, 'target_color.png'))):
                os.remove(join(self.icp_path, 'target_color.png'))
                os.remove(join(self.icp_path, 'target_depth.png'))
            cv2.imwrite(join(self.icp_path, 'target_color.png'), hypo_c.astype(np.uint8))
            cv2.imwrite(join(self.icp_path, 'target_depth.png'), hypo_d.astype(np.uint16))
            f = open(join(self.icp_path, 'ready.txt'), 'w')
            f.write('ready!')
            f.close()
            while not os.path.exists(join(self.icp_path, 'exclusioned.txt')):
                pass
            while not os.path.exists(join(self.icp_path, 'exclusion.txt')):
                pass
            while True:
                try:
                    with open(join(self.icp_path, 'exclusion.txt'), 'r') as f:
                        motion_err = f.readlines()
                        r_err = float(motion_err[0].rstrip('\n'))
                        t_err = float(motion_err[1].rstrip('\n'))
                        motion_err = [r_err, t_err]
                        break
                except:
                    time.sleep(0.0001)
                    print(join(self.icp_path, 'exclusion.txt'))
            os.remove(join(self.icp_path, 'exclusioned.txt'))
            os.remove(join(self.icp_path, 'exclusion.txt'))

            return np.sum(motion_err)


    def cam_succeed(self, pred_blcam):
        hypos = pred_blcam.copy()
        
        c = self.cur_color.astype(np.uint8)  # uint8 [0, 255]
        if self.color_jitter:
            if self.print_color_jitter:
                print(f"\nColor Jittering\n")
                self.print_color_jitter = False
                # color jittering
                c = Image.fromarray(c)
                if np.random.rand() < 0.5:
                    c = tfs.ColorJitter(brightness=.7)(c)
                if np.random.rand() < 0.5:
                    c = tfs.ColorJitter(contrast=.7)(c)
                if np.random.rand() < 0.5:
                    c = tfs.ColorJitter(saturation=.7)(c)

        if (os.path.exists(join(self.icp_path, 'source_color.png'))):
            os.remove(join(self.icp_path, 'source_color.png'))
            os.remove(join(self.icp_path, 'source_depth.png'))
        cv2.imwrite(join(self.icp_path, 'source_color.png'), self.cur_color.astype(np.uint8))
        cv2.imwrite(join(self.icp_path, 'source_depth.png'), self.cur_depth.astype(np.uint16))

        hypo_c, hypo_d = igibson_mesh_renderer.mesh_render(self.mesh_renderer, hypos[0], hidden = (self.mesh_renderer.instances[0], ))

        if (os.path.exists(join(self.icp_path, 'target_color.png'))):
            os.remove(join(self.icp_path, 'target_color.png'))
            os.remove(join(self.icp_path, 'target_depth.png'))
        cv2.imwrite(join(self.icp_path, 'target_color.png'), hypo_c.astype(np.uint8))
        cv2.imwrite(join(self.icp_path, 'target_depth.png'), hypo_d.astype(np.uint16))
        f = open(join(self.icp_path, 'ready.txt'), 'w')
        f.write('ready!')
        f.close()

        while not os.path.exists(join(self.icp_path, 'exclusioned.txt')):
            pass
        while not os.path.exists(join(self.icp_path, 'exclusion.txt')):
            pass

        with open(join(self.icp_path, 'exclusion.txt'), 'r') as f:
            motion_err = f.readlines()
            r_err = float(motion_err[0].rstrip('\n'))
            t_err = float(motion_err[1].rstrip('\n'))
            motion_err = [r_err, t_err]
        self.refresh = 0
        self.motion_err = motion_err
        if motion_err[0] < self.finish[0] and motion_err[1] < self.finish[1]:
            return True

        return False

    def close(self):
        super(EnvRelocUncertainty, self).close()
        pobj = psutil.Process(self.icp.pid)
        for c in pobj.children(recursive=True):
            c.terminate()



def KDtree_search(query, support):
    n = query.shape[0]
    m = support.shape[0]
    tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(support)
    distance, indices = tree.kneighbors(query)  
    return distance.squeeze(-1), indices.squeeze(-1).astype('int64')

def get_colormap(cmap):
    m = matplotlib.cm.ScalarMappable(cmap=cmap)
    a = np.linspace(0, 1, 256)
    c = m.to_rgba(a, bytes=True, norm=False, alpha=False)[:, :-1]
    c = c[:, ::-1]
    colormap = -1 * np.ones((256, 256, 256))
    for i in range(len(a)):
        colormap[c[i][0]][c[i][1]][c[i][2]] = i
    return colormap

def jet_to_number(colormap, image):
    output = np.zeros((image.shape[0], image.shape[1], 1))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i][j][0] = colormap[image[i][j][0]][image[i][j][1]][image[i][j][2]]
    return output