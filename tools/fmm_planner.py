import cv2
import numpy as np
import skfmm
from numpy import ma
import math

###
# This file implement a fast marching algorithm using skfmm, which need a height map to initialize.
###


num_rots = 36

def get_mask(sx, sy, scale, step_size, ori):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    rmask = np.zeros((size, size))
    sx = (size // 2 + sx)
    sy = (size // 2 + sy)
    ori = ori % 360.0
    if ori > 180:
        ori -= 360
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5) - sx
            dy = (j + 0.5) - sy
            if dx ** 2 + dy ** 2 <= step_size ** 2:
                mask[i, j] = 1
                dori = (ori - math.degrees(math.atan2(dx, dy))) % 360.0
                dori = 1.0 - abs(dori / 180.0 - 1.0)
                rmask[i, j] = dori * 1.
    rmask[size // 2, size // 2] = 0.
    return mask, rmask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2:
                mask[i, j] = max(5, (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                     ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner():
    def __init__(self, traversible, num_rots, scale=1, step_size=5):
        self.scale = scale
        self.step_size = step_size
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale, traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.angle_value = [0, 2.0 * np.pi / num_rots, -2.0 * np.pi / num_rots, 0]
        self.du = int(self.step_size / (self.scale * 1.))
        self.num_rots = num_rots

    def set_goal(self, goal):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.)), \
                         int(goal[1] / (self.scale * 1.))
        traversible_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return dd_mask

    def set_goal_by_map(self, map):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        traversible_ma[map == 0] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return dd_mask


    def reachable(self, state):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        traversible_ma[0, 0] = 0
        traversible_ma[0, -1] = 0
        traversible_ma[-1, 0] = 0
        traversible_ma[-1, -1] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        return dd_mask[state[0], state[1]]


    def get_short_term_goal(self, state, ori):
        scale = self.scale * 1.
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask, rmask = get_mask(dx, dy, scale, self.step_size, ori)

        state = [int(x) for x in state]

        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                 state[1]:state[1] + 2 * self.du + 1]


        assert subset.shape[0] == 2 * self.du + 1 and \
            subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset = subset * mask + rmask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2
        subset -= subset[self.du, self.du]

        trav = np.pad(self.traversible, self.du,
                      'constant', constant_values=0)

        subset_trav = trav[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]
        traversible_ma = ma.masked_values(subset_trav * 1, 0)
        goal_x, goal_y = self.du, self.du
        traversible_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        subset_fmm_dist = dd

        subset_fmm_dist[subset_fmm_dist < 4] = 4.
        subset = subset / subset_fmm_dist
        # subset[subset < -1.5] = 1
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        self.debug = (subset, rmask)

        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False
        return (stg_x + state[0] - self.du) * scale + 0.5, \
               (stg_y + state[1] - self.du) * scale + 0.5, replan

