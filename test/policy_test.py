import os
import os.path as osp
import cv2
import torch
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import numpy as np
from rlpyt.envs.gym import make as gym_make
from tools.agent import PPOAgent
from tools import heuristic_action
from tools import fmm_planner
import utils
import global_setting as gls
import time
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import yaml

###
# test one scene with a given setting
###


data_root = gls.data_root
m = matplotlib.cm.ScalarMappable(cmap='BuPu')
grid_size = 7

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


def test(exp_name, num_episodes, cuda_idx, scene_name, seq_name, pretrained_model, net_type, env, vistxt, mode, cfg):
    os.makedirs(osp.join(gls.codes_root, 'vis', mode, scene_name), exist_ok=True)
    env = 'gym_foo:' + env
    exp_name = f'{scene_name}_{seq_name}_' + exp_name
    cfg['scene_names'] = [scene_name]
    cfg['seq_names'] = [seq_name]
    cfg['fixed_pose'] = True
    env = gym_make(env, **cfg, name=exp_name, scene_idx=0, cpp_cuda=cuda_idx, render_cuda=cuda_idx)

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    step_succ = [0 for i in range(10)]

    n = 0
    n_success = 0
    n_cam_success = 0
    n_succ_intersation = 0
    n_succ_union = 0
    n_outbounds = 0
    n_timeout = 0
    max_ep_len = 50

    o, r, d, ep_ret, ep_len, a = env.reset(), 0, False, 0, 0, np.array([0])

    if vistxt:
        f = open(osp.join(gls.codes_root, 'vis', 'mode', scene_name, f'{n}.txt'), 'w')
        action_hist = []
        cam = env.true_cam
        for i in range(len(cam)):
            f.write(str(cam[i]))
            f.write(' ')
        f.write('0\n')


    if mode == 'rl':
        get_rl_action = load_policy(pretrained_model, env, net_type, cuda_idx, mode)
    elif mode == 'turn':
        turn_step = 0
        best_err = np.array(env.motion_err).max()        
        best_turn_step = 0
        s = env.succeed(env.hypo[0], env.true_cam)
    elif mode == 'descent':
        traversable_map = env.traversable_map / 255
        traversible_ma = np.ma.masked_values(traversable_map * 1, 0)
        fmmPlanner = fmm_planner.FMMPlanner(traversable_map, 12, 1, 20)
        errmap_highres = cv2.imread(osp.join(gls.data_root, scene_name, 'errmap', f'{seq_name}_20cm_errmap_avg.png'), -1)
        colormap = get_colormap('jet')
        errmap_highres = jet_to_number(colormap, errmap_highres)
        errmap_highres = env.errmap_highres.squeeze(-1).copy()
        errmap_highres[errmap_highres != -1] = 255 - errmap_highres[errmap_highres != -1]
        traversible_ma[errmap_highres >= 127.5] = 0
        try:
            fmmPlanner.set_goal_by_map(traversible_ma)
        except:
            pass
        with open(osp.join(data_root, scene_name, 'mesh', 'floor_trav_0.yaml'), 'r') as y:
            data = yaml.safe_load(y)
        origin = data['origin']
        assert origin[0] == origin[1], "[floorplan yaml] origin[0] != origin[1]"
        trav_ori = origin[0]

    outbounds = False
    timeout = False
    v_s = False
    policy_time = 0
    step_time = 0
    sum_len = 0
    succ_len = 0
    n_step = 0
    best_err = np.array(env.motion_err).max()
    # --- Acting ---
    while n < num_episodes:
        ta = time.time()
        if mode == 'none':
            d = True
            s = env.succeed(env.hypo[0], env.true_cam)
            outbounds = False
            timeout = False
            err = np.array(env.motion_err).max()        
        elif mode == 'turn':
            if turn_step < 12:
                turn_step += 1
                a = np.array([1])
                o, r, _, info = env.step(a.squeeze())
                err = np.array(env.motion_err).max()
                if err < best_err:
                    best_err = err
                    s = info.succ
                    best_turn_step = turn_step
            else:
                d = True
                extra_len = best_turn_step if best_turn_step <= 6 else 12 - best_turn_step
                ep_len = 12 + extra_len
                outbounds = False
                timeout = False
        else:
            if mode == 'rl':
                a = get_rl_action(o, a, r)
            elif mode == 'random':
                a = env.action_space.sample()
                a = np.asarray(a)
            elif mode == 'descent':
                a = heuristic_action.heu_action_err_from_fp(fmmPlanner, env.state[0], trav_ori)

            if vistxt:
                action_hist.append(a)
            policy_time += time.time() - ta
            tb = time.time()
            o, r, d, info = env.step(a.squeeze())
            step_time += time.time() - tb
            s = info.succ
            outbounds = info.outbounds
            timeout = info.timeout
            ep_ret += r
            ep_len += 1
            n_step += 1


        if not d and not (ep_len == max_ep_len) and vistxt:
            cam = env.true_cam
            for i in range(len(cam)):
                f.write(str(cam[i]))
                f.write(' ')
            f.write('0\n')


        if d or (ep_len == max_ep_len):
            # * to end an episode
            n += 1
            if v_s:
                n_cam_success += 1
            if s:
                n_success += 1
            if s and v_s:
                n_succ_intersation += 1
            if s or v_s:
                n_succ_union += 1
            # count failure cases
            n_outbounds += outbounds
            n_timeout += timeout
            # only store ep_len when succeeded
            if s:
                once_succeed = True
                succ_len += ep_len
                par = int(ep_len) // 10
                if par < 10:
                    step_succ[par] += 1
                cam = env.true_cam
                if vistxt:
                    for i in range(len(cam)):
                        f.write(str(cam[i]))
                        f.write(' ')
                    f.write('1\n')
            else:
                if vistxt:
                    cam = env.true_cam
                    for i in range(len(cam)):
                        f.write(str(cam[i]))
                        f.write(' ')
                    f.write('-1\n')

            sum_len += ep_len

            if vistxt:
                f.close()  

            o, r, d, ep_ret, ep_len, a = env.reset(), 0, False, 0, 0, np.array([0])
            
            if vistxt:
                f = open(osp.join(gls.codes_root, 'vis', 'mode', scene_name, f'{n + 1}.txt'), 'w')
                action_hist = []
                cam = env.true_cam
                for i in range(len(cam)):
                    f.write(str(cam[i]))
                    f.write(' ')
                f.write('0\n')


    succ_ratio = n_success / num_episodes
    cam_succ_ratio = n_cam_success / num_episodes
    if n_succ_union != 0:
        succ_iou = n_succ_intersation / n_succ_union
    else:
        succ_iou = 0
    timeout_ratio = n_timeout / num_episodes
    if (n_success == 0):
        EpLen_succ_avg = 0
    else:
        EpLen_succ_avg = succ_len / n_success
    EpLen_avg = sum_len / num_episodes
    policy_time_avg = policy_time / n_step
    step_time_avg = step_time / n_step

    output = {
        'succ_ratio':succ_ratio, 
        'cam_succ_ratio':cam_succ_ratio, 
        'succ_iou':succ_iou, 
        'timeout_ratio':timeout_ratio, 
        'EpLen_succ_avg':EpLen_succ_avg, 
        'EpLen_avg':EpLen_avg, 
        'policy_time_avg':policy_time_avg, 
        'step_time_avg':step_time_avg
    }
    for i in range(10):
        output[f'{i * 10}steps_succ_ratio'] = step_succ[i] / num_episodes

    for key in output:
        print(f'{key}:\t{output[key]}')

    f.close()
    env.close()


def load_policy(pretrained_model, env, net_type, cuda_idx, greedy_eval=False):
    print(f'Loading pretrained RL model from {pretrained_model}')
    initial_model_state_dict = torch.load(pretrained_model, map_location=f'cuda:{cuda_idx}')['agent_state_dict']

    agent = PPOAgent(greedy_eval=greedy_eval, initial_model_state_dict=initial_model_state_dict,
                             model_kwargs=dict(net_type=net_type))
    agent.initialize(env.spaces)
    agent.eval_mode(0)

    def get_rl_action(obs, action, rew):
        action, action_info = agent.step(torch.from_numpy(obs).float(),
                                         torch.from_numpy(action).float(),
                                         torch.tensor(rew).float())
        action = action.numpy()
        return action
    
    return get_rl_action

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', required=True, type=str)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--net-type', required=True, type=str)
    parser.add_argument('--env', type=str, default='EnvRelocUncertainty-v0')
    parser.add_argument('--scene-name', required=True, type=str)
    parser.add_argument('--seq-name', required=True, type=str)
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--cuda-idx', required=True, type=int)
    parser.add_argument('--vistxt', action='store_true')
    parser.add_argument('--mode', type=str, default='rl')
    parser.add_argument('--cfg', type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as y:
        cfg = yaml.safe_load(y)

    test(args.exp_name, args.episodes, args.cuda_idx, args.scene_name, args.seq_name, args.ckpt, args.net_type, args.env, args.vistxt, args.mode, cfg)
