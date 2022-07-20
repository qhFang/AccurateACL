import numpy as np
import math
import cv2
import scipy.io


def pose2blcam(pose):
    """
    :param pose: camera to world
    :return: blender camera [x, y, z, rx, ry, rz]
    """
    R_blcam2cv = np.array(
        [[1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]])
    pose_bl = pose @ R_blcam2cv
    t_bl = pose_bl[:3, 3]
    r_bl = rotationMatrixToEulerAngles(pose_bl[:3, :3])
    cam_bl = np.hstack((t_bl, r_bl))
    return cam_bl


def blcam2pose(blcam):
    R_blcam2cv = np.array(
        [[1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]])
    R_bl = eulerAnglesToRotationMatrix(blcam[3:])
    pose_bl = np.eye(4)
    pose_bl[:3, :3] = R_bl
    pose_bl[:3, 3] = blcam[:3]
    pose = pose_bl @ R_blcam2cv
    return pose


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    # euler order 'XYZ'
    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(theta):
    """ Calculates Rotation Matrix given 'XYZ' euler angles. """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])
    # euler order 'XYZ'
    R = R_z @ R_y @ R_x
    return R


def rt_err(pose1, pose2):
    # t err
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]
    t_err = np.linalg.norm((t1 - t2))
    t_err *= 100
    # r_err
    r1 = pose1[:3, :3]
    r2 = pose2[:3, :3]
    r_err_m = r1 @ np.linalg.inv(r2)
    r_err_v = cv2.Rodrigues(r_err_m)[0].squeeze()
    r_err = np.linalg.norm(r_err_v)
    r_err = np.rad2deg(r_err)
    return r_err, t_err


def rot_x(rad):
    mat = np.array([[1, 0, 0],
                    [0, math.cos(rad), -math.sin(rad)],
                    [0, math.sin(rad), math.cos(rad)],
                    ])
    return mat


def rot_y(rad):
    mat = np.array([[math.cos(rad), 0, math.sin(rad)],
                    [0, 1, 0],
                    [-math.sin(rad), 0, math.cos(rad)],
                    ])
    return mat


def rot_z(rad):
    mat = np.array([[math.cos(rad), -math.sin(rad), 0],
                    [math.sin(rad), math.cos(rad), 0],
                    [0, 0, 1],
                    ])
    return mat


def blcam2travxy(ori, bl_x, bl_y):
    img_coord = (int(round((-bl_y - ori) * 100)), int(round((bl_x - ori) * 100)))
    return img_coord

def batch_blcam2travxy(ori, bl_xy):
    #img_coord = np.zeros((bl_xy.shape[0], 2))
    #for i in range(bl_xy.shape[0]):
    #    img_coord[0], img_coord[1] = blcam2travxy(ori, bl_xy[i][0], bl_xy[i][1])
    #img_coord = img_coord.astype(np.int64)
    bl_xy[:, 1] = -bl_xy[:, 1]
    img_coord = np.round((np.flip(bl_xy, 1) - ori) * 100).astype(np.int64)
    return img_coord


def travxy2blcam(ori, trav_x, trav_y):
    trav_x, trav_y = float(trav_x), float(trav_y)
    blcam = (trav_y / 100 + ori, -trav_x / 100 - ori)
    return blcam


def read_gt(datalist):
    """
    :param datalist: training sequence list
    :return: blender camera of size of (N, 6)
    """
    blcam_train = np.zeros((0, 6))
    with open(datalist, 'r') as file:
        for f in file:
            f = f.strip()
            f = f.replace('depth.png', 'pose.txt')
            pose = np.loadtxt(f)
            blcam = pose2blcam(pose)
            blcam_train = np.vstack((blcam_train, blcam[np.newaxis]))
    return blcam_train


def env_kwargs_from_name(exp_name):
    compo = exp_name.split('_')
    env_kwargs = dict(
        name=exp_name,
        scene_names=list(compo[0].split('#')),
        seq_names=list(compo[1].split('#')),
        noise_std=0,
        step_size=0.1,
        angle_size=30,
        var_th=1e-4,
        var_rew=False,
        n_hyp=4,
        color_jitter=False,
        wocollide = False,
        special_pose = False,
    )
    for c in compo[2:]:
        if 'noise' in c:
            env_kwargs['noise_std'] = float(c.replace('noise', '')) / 100
        elif ('cm' in c) and ('succ' not in c):
            env_kwargs['step_size'] = float(c.replace('cm', '')) / 100
        elif ('deg' in c) and ('succ' not in c):
            env_kwargs['angle_size'] = float(c.replace('deg', ''))
        elif 'succrew' in c:
            env_kwargs['success_reward'] = float(c.replace('succrew', ''))
        elif 'nhyp' in c:
            env_kwargs['n_hyp'] = int(c.replace('nhyp', ''))
        elif 'colorj' in c:
            env_kwargs['color_jitter'] = True
        elif 'succcm' in c:
            env_kwargs['succ_cm'] = int(c.replace('succcm', ''))
        elif 'succdeg' in c:
            env_kwargs['succ_deg'] = int(c.replace('succdeg', ''))
        elif 'timep' in c:
            env_kwargs['time_punish'] = int(c.replace('timep', ''))
        elif 'wocollide' in c:
            env_kwargs['wocollide'] = True
        elif 'finish' in c:
            env_kwargs['finish_condition'] = int(c.replace('finish', ''))

    return env_kwargs


def array2mat(arr):
    m = dict(mat=arr)
    scipy.io.savemat('gt.mat', m)

def get_pcd_from_depth_and_camera(depth, intrinsic, extrinsic, img_size=None, wobody=False):
    d = depth / 1e3 # depth as meters

    gridy, gridx = np.mgrid[:480, :640]
    x_cam = (gridx - intrinsic[0, 2]) / intrinsic[0, 0] * d
    y_cam = (gridy - intrinsic[1, 2]) / intrinsic[1, 1] * d
    
    assert d.shape == x_cam.shape

    if img_size is not None:
        x_cam = cv2.resize(x_cam, (img_size, img_size))
        y_cam = cv2.resize(y_cam, (img_size, img_size))
        d = cv2.resize(d, (img_size, img_size))
    
    pose = blcam2pose(extrinsic)
    if wobody:        
        x_cam, y_cam, d = x_cam[x_cam.shape[0] // 2], y_cam[y_cam.shape[0] // 2], d[d.shape[0] // 2]
        xyz = np.concatenate((x_cam[..., np.newaxis], y_cam[..., np.newaxis], d[..., np.newaxis]), axis=1)
    else:
        xyz = np.concatenate((x_cam[..., np.newaxis], y_cam[..., np.newaxis], d[..., np.newaxis]), axis=2)
        xyz = xyz.reshape((xyz.shape[0] * xyz.shape[1], 3))
    tmp_column = np.ones(xyz.shape[0])
    xyz = np.c_[xyz, tmp_column].copy()
    xyz = np.dot(pose, xyz.T).T[:, :3]
    np.savetxt('/apdcephfs/private_qihangfang/111.txt', xyz)
    return xyz

def draw_lines(map, start_point, end_points):
    assert len(start_point.shape) == 1  and (len(end_points.shape) == 1 or len(end_points.shape) == 2) 
    
    X = 0
    Y = 1
    map[int(start_point[X]), int(start_point[Y])] = 1

    if len(end_points.shape) == 1:
        end_points = end_points.reshape((1, end_points.shape[0]))
    
    for i in range(len(end_points)):
        grid_start, grid_end = (start_point[1], start_point[0]), (end_points[i, 1], end_points[i, 0])
        cv2.line(map, grid_start, grid_end, 1.0)
        '''
        if grid_start[X] == grid_end[X] and grid_start[Y] == grid_end[Y]:
            continue

        # Check the sliding dim 
        dx = abs(grid_end[X] - grid_start[X])
        dy = abs(grid_end[Y] - grid_start[Y])
        dim = X
        if dx != 0:
            delta = dy / dx
        if (dy > dx):
            dim = Y
            delta = dx / dy

        # Make sure that the line is draw from lower one to higher one along the slinding dim
        if grid_start[X] > grid_end[X] and dim == X:
            grid_start, grid_end = grid_end, grid_start
        elif grid_start[Y] > grid_end[Y] and dim == Y:
            grid_start, grid_end = grid_end, grid_start
        
        # Check the line is draw from lower to higher along the other dim or not 
        flag = 1 if grid_end[dim ^ 1] > grid_start[dim ^ 1] else -1

        grid = [grid_start[X], grid_start[Y]]
        acc_delta = 0
        for j in range(grid_start[dim] + 1, grid_end[dim]):
            print(grid)
            grid[dim] += 1
            acc_delta += delta
            if (acc_delta >= 1):
                grid[dim ^ 1] += flag
                acc_delta = 0
            map[int(grid[X]), int(grid[Y])] = 1
        exit()
        '''
    return map





if __name__ == '__main__':
    print(blcam2travxy(-9, 2, 3))
    print(travxy2blcam(-9, 600, 1100))
