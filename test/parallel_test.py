import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))
import policy_test
import yaml

###
# test muiltiply scenes at the same time
###

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', required=True, type=str)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--net-type', required=True, type=str)
    parser.add_argument('--env', type=str, default='EnvRelocUncertainty-v0')
    parser.add_argument('--scene-names', required=True, type=str)
    parser.add_argument('--seq-names', required=True, type=str)
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--vistxt', action='store_true')
    parser.add_argument('--mode', type=str, default='rl')
    parser.add_argument('--cfg', type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as y:
        cfg = yaml.safe_load(y)

    scene_names = args.scene_names.split('#')
    seq_names = args.seq_names.split('#')
    assert len(scene_names) == len(seq_names), "the number of scenes is not equal to the number of sequences"

    for i in range(len(scene_names)):
        policy_test.test(args.exp_name, args.episodes, i % args.gpu_num, scene_names[i], seq_names[i], args.ckpt, args.net_type, args.env, args.vistxt, args.mode, cfg)
