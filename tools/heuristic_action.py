import numpy as np
import os
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import gym
import skfmm
import fmm_planner
import math

###
# implement of some heuristic actions
###

def heu_action(o, blcam_train):
    """
    6 actions
    :param o: current hypotheses, the hypothesis with highest confidence will be seen as current pose
    :param blcam_train: training sequence used to train the localizer. format: blender camera of size of (N, 6)
    :return: heuristic action
    """
    # hypo_loc = o.mean(0)
    hypo_loc = o[0]
    targ_loc = blcam_train
    dist_loc = np.linalg.norm(hypo_loc - targ_loc, axis=1)
    chosen_loc = targ_loc[np.argmin(dist_loc)]

    action_base = np.array([[1, 0, 0, 0, 0, 0],     # x forward
                            [-1, 0, 0, 0, 0, 0],    # x backward
                            [0, 1, 0, 0, 0, 0],     # y forward
                            [0, -1, 0, 0, 0, 0],    # y backward
                            [0, 0, 0, 0, 0, 1],     # turn left
                            [0, 0, 0, 0, 0, -1],    # turn right
                            ])
    direc_vec = chosen_loc - hypo_loc
    direct_dot = np.dot(action_base, direc_vec)
    action = np.argmax(direct_dot)
    return action


def heu_action_wocollide(o, blcam_train, trav, ori):
    hypo_loc = o[0]
    hypo_x = hypo_loc[0]
    hypo_y = hypo_loc[1]
    hypo_rz = hypo_loc[5] / np.pi * 180

    map_xy = [(-hypo_x - ori) * 100, (hypo_y - ori) * 100]
    if (map_xy[0] < 0):
        map_xy[0] = 0
    if (map_xy[0] >= trav.shape[0]):
        map_xy[0] = trav.shape[0] - 1
    if (map_xy[1] < 0):
        map_xy[1] = 0
    if (map_xy[1] >= trav.shape[1]):
        map_xy[1] = trav.shape[1] - 1

    fmmPlanner = fmm_planner.FMMPlanner(trav, 12, 5, 20)
    fmmPlanner.set_goal(blcam_train)
    short_term_goal_x, short_term_goal_y, _ = fmmPlanner.get_short_term_goal(map_xy, hypo_rz)
    goal = [round(short_term_goal_x), round(short_term_goal_y)]
    (short_term_goal_x, short_term_goal_y) = (short_term_goal_y / 100 + ori, -short_term_goal_x / 100 - ori)

    dx = short_term_goal_x - hypo_x
    dy = short_term_goal_y - hypo_y
    drz = hypo_rz - math.degrees(math.atan2(dx, dy))
    if (abs(drz) >= 30):
        if (drz > 0):
            return (map_xy, goal, np.array([1]))
        else:
            return (map_xy, goal, np.array([2]))
    return (map_xy, goal, np.array([0]))

def heu_action_err(env, blcam_train, ori):
    hypo_loc = env.state[0]
    trav = env.traversable_map
    
    hypo_x = hypo_loc[0]
    hypo_y = hypo_loc[1]
    hypo_rz = hypo_loc[5] / np.pi * 180

    map_xy = [int(-hypo_x - ori) * 100, int(hypo_y - ori) * 100]


def heu_action_err(o, blcam_train, trav, ori):
    hypo_loc = o[0]
    hypo_x = hypo_loc[0]
    hypo_y = hypo_loc[1]
    hypo_rz = hypo_loc[5] / np.pi * 180

    map_xy = [int(-hypo_x - ori) * 100, int(hypo_y - ori) * 100]
    if (map_xy[0] < 0):
        map_xy[0] = 0
    if (map_xy[0] >= trav.shape[0]):
        map_xy[0] = trav.shape[0] - 1
    if (map_xy[1] < 0):
        map_xy[1] = 0
    if (map_xy[1] >= trav.shape[1]):
        map_xy[1] = trav.shape[1] - 1
    map_rz = int(hypo_loc[5] / np.pi * 6) % 12

    for i in range(blcam_train.shape[0]):
        if ([map_xy[0], map_xy[1], map_rz] in [blcam_train[i].tolist(), blcam_train[i].tolist()]):
            return (map_xy, blcam_train[i, :2], np.array([3, i]))
    for i in range(blcam_train.shape[0]):
        if (map_xy in [blcam_train[i, :2].tolist(), blcam_train[i, :2].tolist()]):
            shift_rz = blcam_train[i][2] - map_rz
            if shift_rz > 6:
                shift_rz = -1
            if shift_rz < -6:
                shift_rz = 1
            if (shift_rz > 0):
                return (map_xy, blcam_train[i, :2], np.array([1]))
            else:
                return (map_xy, blcam_train[i, :2], np.array([2]))
    fmmPlanner = fmm_planner.FMMPlanner(trav, 12, 5, 20)
    fmmPlanner.set_goal(blcam_train.T[:, :2])
    short_term_goal_x, short_term_goal_y, _ = fmmPlanner.get_short_term_goal(map_xy, hypo_rz)
    goal = [round(short_term_goal_x), round(short_term_goal_y)]
    (short_term_goal_x, short_term_goal_y) = (short_term_goal_y / 100 + ori, -short_term_goal_x / 100 - ori)

    dx = short_term_goal_x - hypo_x
    dy = short_term_goal_y - hypo_y
    drz = hypo_rz - math.degrees(math.atan2(dx, dy))
    if (abs(drz) >= 30):
        if (drz > 0):
            return (map_xy, goal, np.array([1]))
        else:
            return (map_xy, goal, np.array([2]))
    return (map_xy, goal, np.array([0]))

def heu_action_err_from_fp(fmmPlanner, cam, ori):
    if np.isnan(np.array(cam)).any():
        hypo_x = 0
        hypo_y = 0
        hypo_rz = 0
    else:
        hypo_x = cam[0]
        hypo_y = cam[1]
        hypo_rz = cam[5] / np.pi * 180

    map_xy = [int(-hypo_x - ori) * 100, int(hypo_y - ori) * 100]
    try:
        short_term_goal_x, short_term_goal_y, _ = fmmPlanner.get_short_term_goal(map_xy, hypo_rz)
    except:
        return 0
    goal = [round(short_term_goal_x), round(short_term_goal_y)]
    (short_term_goal_x, short_term_goal_y) = (short_term_goal_y / 100 + ori, -short_term_goal_x / 100 - ori)

    dx = short_term_goal_x - hypo_x
    dy = short_term_goal_y - hypo_y
    drz = hypo_rz - math.degrees(math.atan2(dx, dy))
    if (abs(drz) >= 30):
        if (drz > 0):
            return 1
        else:
            return 2
    return 0

def anticlock_angle(v1, v2):
    """ 3d vector, order matters """
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot / norm)
    cross = np.cross(v1, v2)
    if cross[-1] > 0:
        return angle
    else:
        return -angle


def heu_action_act3(o, blcam_train):
    """
    3 actions
    :param o: current hypotheses, the hypothesis with highest confidence will be seen as current pose
    :param blcam_train: training sequence used to train the localizer. format: blender camera of size of (N, 6)
    :return: heuristic action
    """
    # hypo_loc = o.mean(0)
    hypo_loc = o[0]
    targ_loc = blcam_train
    # only compute location distance
    dist_loc = np.linalg.norm(hypo_loc[:3] - targ_loc[:, :3], axis=1)
    chosen_loc = targ_loc[np.argmin(dist_loc)]

    # turn left or right if the intersection angle more than 30 deg
    direc_vec = chosen_loc - hypo_loc
    direc_vec_loc = direc_vec[:3]
    direc_vec_loc[2] = 0        # only care about (x, y) coordinate
    curr_view = np.array([-np.sin(hypo_loc[-1]), np.cos(hypo_loc[-1]), 0])
    angle = anticlock_angle(curr_view, direc_vec_loc)
    if np.abs(angle) < np.pi / 6 and np.linalg.norm(direc_vec_loc) > 0.15:
        # The agent faces and is not very close to the target
        # forward
        action = 0
    elif angle > 0:
        # turn left
        action = 1
    else:
        # turn right
        action = 2
    action = np.array(action)
    return action


def heu_action_greedy(env):
    """ Try every possible action and choose the one with minimum variance """
    action = -1
    o_var_min = np.inf
    for a in range(env.action_space.n):
        o = env.try_action(a)
        o_var = o
        if o_var < o_var_min:       # robust to nan
            o_var_min = o_var
            action = a
    if action == -1:
        # fail to select an action, just randomly select one
        action = np.random.randint(env.action_space.n)
    action = np.array(action)
    return action

