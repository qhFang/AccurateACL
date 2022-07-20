import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
import sys
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

###
#network for RL algorithm
###

class PPORelocNet(nn.Module):
    def __init__(self, observation_shape, action_size, net_type):
        """Instantiate neural net modules according to inputs."""
        super(PPORelocNet, self).__init__()
        if net_type == "woscene":
            net = RelocNetwoscene
        elif net_type == "wocamera":
            net = RelocNetwocamera
        elif net_type == "uncertainty":
            net = RelocNetUncertainty
        else:
            raise ValueError(f'invalid network type: {net_type}')
        self._obs_ndim = len(observation_shape)
        self.pi = net(out_dim=action_size, )
        self.v = net(out_dim=1, )

    def forward(self, observation, prev_action, prev_reward):
        """
        :param observation: (B, *observation_size)
        :param prev_action: one-hot (B, action_n)
        :param prev_reward: (B)
        :return:
        """
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        # obs_flat = observation.view(T * B, -1)
        if lead_dim == 0:
            observation = torch.unsqueeze(observation, 0)
        pi = F.softmax(self.pi(observation), dim=-1)
        v = self.v(observation).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v

    def set_device(self, device):
        self.device = device


class RelocNetUncertainty(nn.Module):
    """Current view, current point cloud, seq (partial scene) point cloud """
    def __init__(self, out_dim, encoder_only=False):
        super(RelocNetUncertainty, self).__init__()
        self.vgg_2d = _vgg(None, 'K', True, False, True, num_classes=64, in_channels=3)
        self.pcn_3d = PointNet(n_channel_in=6, n_channel_out=64)
        self.mlp_uncertainty = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64))
        self.post = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, out_dim))
        self.encoder_only = encoder_only

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)
        batch_size = inputs.shape[0]
        
        inputs_2d = inputs[:, :48].reshape(batch_size, 3, 256, 256)
        inputs_3d = inputs[:, 48:72].reshape(batch_size, 6, 2**14)
        inputs_uncertainty = inputs[:, 72, :2, 0].reshape(batch_size, 2)
                
        o_2d = self.vgg_2d(inputs_2d)
        o_3d = self.pcn_3d(inputs_3d)
        o_uncertainty = self.mlp_uncertainty(inputs_uncertainty)
        o = torch.cat((o_2d, o_3d, o_uncertainty), 1)

        if self.encoder_only:
            return o
        else:
            o = self.post(o)
            return o

class RelocNetwoscene(nn.Module):
    """Current view, current point cloud, seq (partial scene) point cloud """
    def __init__(self, out_dim, encoder_only=False):
        super(RelocNetwoscene, self).__init__()
        self.vgg_2d = _vgg(None, 'K', True, False, True, num_classes=64, in_channels=3)
        self.mlp_uncertainty = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64))
        self.post = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, out_dim))
        self.encoder_only = encoder_only

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)
        batch_size = inputs.shape[0]
        
        inputs_2d = inputs[:, :48].reshape(batch_size, 3, 256, 256)
        inputs_uncertainty = inputs[:, 72, :2, 0].reshape(batch_size, 2)

                
        o_2d = self.vgg_2d(inputs_2d)
        o_uncertainty = self.mlp_uncertainty(inputs_uncertainty)
        o = torch.cat((o_2d, o_uncertainty), 1)

        if self.encoder_only:
            return o
        else:
            o = self.post(o)
            return o

class RelocNetwocamera(nn.Module):
    """Current view, current point cloud, seq (partial scene) point cloud """
    def __init__(self, out_dim, encoder_only=False):
        super(RelocNetwocamera, self).__init__()
        self.pcn_3d = PointNet(n_channel_in=6, n_channel_out=64)
        self.mlp_uncertainty = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64))
        self.post = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, out_dim))
        self.encoder_only = encoder_only

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)
        batch_size = inputs.shape[0]
        
        inputs_3d = inputs[:, 48:72].reshape(batch_size, 6, 2**14)
        inputs_uncertainty = inputs[:, 72, :2, 0].reshape(batch_size, 2)

        o_3d = self.pcn_3d(inputs_3d)
        o_uncertainty = self.mlp_uncertainty(inputs_uncertainty)
        o = torch.cat((o_3d, o_uncertainty), 1)

        if self.encoder_only:
            return o
        else:
            o = self.post(o)
            return o

class PointNet(nn.Module):
    def __init__(self, n_channel_out=128, n_group=1):
        super(PointNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.GroupNorm(n_group, 64), # GN
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(n_group, 128), # GN
            nn.ReLU(inplace=True),
            nn.Conv1d(128, n_channel_out, 1)
        )
    def __init__(self, n_channel_in, n_channel_out=128, n_group=1):
        super(PointNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(n_channel_in, 64, 1),
            nn.GroupNorm(n_group, 64), # GN
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(n_group, 128), # GN
            nn.ReLU(inplace=True),
            nn.Conv1d(128, n_channel_out, 1)
        )
    def forward(self, x): # x: (BS, channel, N)
        x = self.mlp(x)
        x = torch.max(x, 2)[0]
        return x


def pixel2camera(depth, color=None):
    """
    :param depth: [h, w] for single image, [batch, h, w] for batched images
    :return: [N, 3] for single depth image, [batch, N, 3] for batched depth images
    """
    if depth.ndim == 3:
        batch, depth_h, depth_w = depth.shape
    else:
        batch = None
        depth_h, depth_w = depth.shape
    cx = depth_w / 2
    cy = depth_h / 2
    fx = 0.75 * depth_w
    fy = depth_h
    u_base = torch.Tensor(np.tile(np.arange(depth_w)[np.newaxis, :], (depth_h, 1))).to(depth.device)
    v_base = torch.Tensor(np.tile(np.arange(depth_h)[:, np.newaxis], (1, depth_w))).to(depth.device)
    X = (u_base - cx) * depth / fx
    Y = (v_base - cy) * depth / fy
    coord_camera = torch.stack((X, Y, depth), axis=-1)
    if not batch:
        points_camera = coord_camera.reshape((-1, 3))	        # (N, 3)
        points_camera = points_camera.permute((0, 2, 1))        # (3, N)
    else:
        points_camera = coord_camera.reshape((batch, -1, 3))    # (b, N, 3)
        points_camera = points_camera.permute((0, 2, 1))        # (b, 3, N)
    if color is None:
        return points_camera
    else:
        color = color.permute((0, 2, 3, 1))
        if not batch:
            points_color = color.reshape((-1, 3))			    # (N, 3)
            points_color = points_color.transpose(1, 0)         # (3, N)
        else:
            points_color = color.reshape((batch, -1, 3))        # (b, N, 3)
            points_color = points_color.permute((0, 2, 1))      # (b, 3, N)
        points_xyzbgr = torch.cat((points_camera, points_color), dim=-2)
        return points_xyzbgr


class VGG(nn.Module):
    def __init__(self, features, classifier=None, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, num_classes),
            )
        else:
            self.classifier = classifier
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'I': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256, 'M'],
    'J': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'H': [32, 'M', 64, 'M', 128, 'M', 256, 'M'],
    'K': [32, 'M', 64, 'M', 128, 'M', 128, 'M', 256, 'M', 256, 'M'],

}


def _vgg(arch, cfg, batch_norm, pretrained, progress, in_channels, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, in_channels=in_channels), **kwargs)
    if pretrained:
        raise NotImplementedError
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
