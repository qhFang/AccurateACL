import datetime
import os
from os import path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'tools'))
import argparse
import torch
from torch import nn
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.algos.pg.ppo import *
from rlpyt.envs.gym import make as gym_make
from agent import PPOAgent, PPOLSTMAgent
from env import RelocTrajInfo
import psutil
import yaml
import utils
import multiprocessing as mp
 

def build_and_train(exp_name, env_id, net_type='base', run_id=0, greedy_eval=False, pretrained_model=None,
                    lr=3e-4,
                    batch_T=800,
                    batch_B=6,
                    batch_nn=40,
                    cuda_idx=None,
                    cuda_idx_cpp=None,
                    cuda_idx_render=None,
                    cfg=None,
                    LOG_DIR='ckpt',
                    ):
    log_name = osp.join(LOG_DIR, exp_name, datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
    if len(log_name) > 200:
        log_dir = osp.join(LOG_DIR, 'mutiply', datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
    else:
        log_dir = log_name
    n_steps = int(500 * batch_T * batch_B)
    minibatches = batch_T * batch_B // batch_nn
    log_step = 1
    with open(cfg, 'r') as y:
        cfg = yaml.safe_load(y)
    cfg['scene_names'] = cfg['scene_names'].split('#')
    cfg['seq_names'] = cfg['seq_names'].split('#')
    env_kwargs = dict(
        name=exp_name, id=env_id, **cfg, cpp_cuda=cuda_idx_cpp, render_cuda=cuda_idx_render,
    )
    sampler_kwargs = dict(
        batch_T=batch_T,
        batch_B=batch_B,
    )
    agent_kwargs = dict(
        model_kwargs=dict(
            net_type=net_type,
        )
    )
    algo_kwargs = dict(
        learning_rate=lr,
        minibatches=minibatches,
        entropy_loss_coeff=0.,

        discount=0.99,
        value_loss_coeff=1.,
        clip_grad_norm=1.,
        gae_lambda=1,
        epochs=4,
        ratio_clip=0.1,
        linear_lr_schedule=True,
        normalize_advantage=False,
    )
    p = psutil.Process()
    cpu_affin = p.cpu_affinity()
    affinity = dict(
        cuda_idx=cuda_idx,
        workers_cpus=cpu_affin[:batch_B],
    )

    if pretrained_model:
        print(f'Load pretrained model from {pretrained_model}')
        model = torch.load(pretrained_model)
        model_agent = model['agent_state_dict']
        model_optim = model['optimizer_state_dict']
        agent_kwargs['initial_model_state_dict'] = model_agent
        algo_kwargs['initial_optim_state_dict'] = model_optim

    sampler = GpuSampler(
        EnvCls=gym_make,
        env_kwargs=env_kwargs,
        TrajInfoCls=RelocTrajInfo,
        max_decorrelation_steps=0,      # take n random steps, should be 0
        **sampler_kwargs,
    )
    algo = PPO(
        **algo_kwargs,
    )
    if 'lstm' in exp_name:
        agent = PPOLSTMAgent(
            greedy_eval,
            **agent_kwargs,
        )
    else:
        agent = PPOAgent(
            greedy_eval,
            **agent_kwargs,
        )

    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=n_steps,
        log_interval_steps=log_step * int(batch_B * batch_T),
        affinity=affinity,
    )

    # hyper parameters
    log_params = dict(
        code=osp.abspath(__file__),
        exp_name=exp_name,
        env_id=env_id,
        log_step=log_step,
        **sampler_kwargs,
        nn_batchsize=batch_T * batch_B // minibatches,
        **algo_kwargs,
        **env_kwargs,
    )

    print(f'snapshot mode: {args.snapshot_mode}')
    with logger_context(log_dir, run_id, name=log_name, log_params=log_params, snapshot_mode=args.snapshot_mode,
                        use_summary_writer=True, override_prefix=True):
        runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_name', type=str, default='test_left_online_var_th1e-4_cont_noise5_10cm_act3', help='rf_path, checkpoints (wrapped by time), specify several params')
    parser.add_argument('--env', type=str, default='EnvReloc-v0')
    parser.add_argument('--net_type', type=str, default='base', help='neural network to be used')
    parser.add_argument('--batch_B', type=int, default=6, help='how many envs to execute')
    parser.add_argument('--batch_T', type=int, default=800, help='how many time steps to execute')
    parser.add_argument('--batch_nn', type=int, default=40, help='minibatch size')
    parser.add_argument('--run_id', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_cpp', type=int, default=0)
    parser.add_argument('--gpu_render', type=int, default=0)
    parser.add_argument('--pretrained_model', type=str, help='absolute path of pretrained model')
    parser.add_argument('--snapshot_mode', type=str, default='last', choices=['last', 'all'])
    parser.add_argument('--cfg', type=str, default='configs/train.yaml', help='the path of config file')
    parser.add_argument('--log_dir', type=str, default='ckpt', help='the path of config file')

    args = parser.parse_args()
    args.env = 'gym_foo:' + args.env
    build_and_train(exp_name=args.exp_name, 
                    env_id=args.env, 
                    batch_B=args.batch_B, 
                    batch_T=args.batch_T, 
                    batch_nn=args.batch_nn, 
                    net_type=args.net_type, 
                    run_id=args.run_id, 
                    cuda_idx=args.gpu, 
                    cuda_idx_cpp=args.gpu_cpp, 
                    cuda_idx_render=args.gpu_render, 
                    pretrained_model=args.pretrained_model, 
                    cfg=args.cfg, 
                    LOG_DIR=args.log_dir)
