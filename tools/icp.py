import open3d as o3d
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import numpy as np

###
# This file contains an implementation of colorICP based on open3d and a file connection module.
#
# note that:
#            There may be some unexpected bugs because of the subprocess module.
#            If you find a bug like this, please contact us. 


def execute_color_registration(source, target):
    voxel_radius = [0.16, 0.08, 0.04]
    max_iter = [30, 20, 10]
    current_transformation = np.identity(4)
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation


    return utils.rt_err(np.identity(4), current_transformation)

def worker(filepath):
    os.remove(os.path.join(filepath, 'ready.txt'))
    color1 = o3d.io.read_image(os.path.join(filepath, 'source_color.png'))
    depth1 = o3d.io.read_image(os.path.join(filepath, 'source_depth.png'))
    rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color1, depth1, depth_scale=1000, convert_rgb_to_intensity=False)
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image1, o3d_intrinsic)

    color2 = o3d.io.read_image(os.path.join(filepath, 'target_color.png'))
    depth2 = o3d.io.read_image(os.path.join(filepath, 'target_depth.png'))
    rgbd_image2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color2, depth2, depth_scale=1000, convert_rgb_to_intensity=False)
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image2, o3d_intrinsic)

    # The target image is rendered from an incomplete reconstructed scene
    # which means it contains only a few points sometimes at which we will set a large error to return.
    try:
        r_err, t_err = execute_color_registration(pcd1,pcd2)
    except:
        r_err, t_err = 100., 100.

    while True:
        try:
            f = open(os.path.join(filepath, 'exclusion.txt'), 'w')
            f.write(str(r_err))
            f.write('\n')
            f.write(str(t_err))
            f.close()
            f = open(os.path.join(filepath, 'exclusioned.txt'), 'w')
            f.close()
            break
        except:
            pass


if __name__ == "__main__":
    filepath = sys.argv[-2]
    worker_id = sys.argv[-1]

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    o3d_intrinsic.set_intrinsics(640, 480, 480, 480, 320, 240)

    while True:
        if os.path.exists(os.path.join(filepath, 'ready.txt')):
            worker(filepath)