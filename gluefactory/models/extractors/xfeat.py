# import os
# from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack


class BasicLayer(nn.Module):
    """
      Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """

    # 初始化插值模式 mode 和是否对齐角点 align_corners
    # 默认情况下，插值模式为 'bicubic'，并且不对齐角点
    def __init__(self, mode='bicubic', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    # 该方法用于将输入的二维位置坐标归一化到 [-1, 1] 的范围内
    # 因为 PyTorch 中的 F.grid_sample 方法要求输入的坐标在 [-1, 1] 范围内
    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x / (torch.tensor([W - 1, H - 1], device=x.device, dtype=x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode=self.mode, align_corners=False)
        return x.permute(0, 2, 3, 1).squeeze(-2)


class XFeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)
        self.skip1 = nn.Sequential(nn.AvgPool2d(4, stride=4), nn.Conv2d(1, 24, 1, stride=1, padding=0))
        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )
        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )
        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )
        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )
        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0)
        )
        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        # Fine Matcher MLP
        # 这里如果不加这个部分，会有未匹配的参数
        self.fine_matcher =  nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 64),
        )

    def _unfold2d(self, x, ws: int = 2):
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(B, C, H // ws, W // ws, ws ** 2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)

        heatmap = self.heatmap_head(feats)
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))

        return feats, keypoints, heatmap


class XFeat(BaseModel):
    default_conf = {
        "descriptor_dim": 256,  # lightglue 需要 256 维度的描述子，而非 64 维
        "nms_radius": 2,  # kernel size = nms_radius * 2 + 1
        "max_num_keypoints": 4096,
        "force_num_keypoints": False,
        "detection_threshold": 0.05,
        "remove_borders": 4,
    }

    def _init(self, conf):
        self.conf = SimpleNamespace(**conf)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.top_k = self.conf.max_num_keypoints
        self.net = XFeatModel().to(self.device).eval()
        self.interpolator = InterpolateSparse2d('bicubic')
        self._load_weights()
        '''
        新增一个线性层，它将描述符扩展到 256 维度
        该线性层用于处理 xfeat 的描述符
        '''
        self.fc = nn.Linear(64, 256)

    def _load_weights(self):
        # weights_path = '/home/leesin/Develop/DeepLearning/glue-factory/weights/xfeat_scripted.pt'
        # 需要加载 state_dict 模型文件而非 torch script 模型文件
        # 对于 state_dict ，它由 model.state_dict() 保存
        # 对于 TorchScript 存档，它由 torch.jit.script 加载
        weights_path = '/home/leesin/Develop/DeepLearning/glue-factory/weights/xfeat.pt'
        self.net.load_state_dict(torch.load(weights_path, map_location=self.device))

    def _forward(self, data):
        # 这里的 x 就是 image
        x = data["image"].to(self.device)
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        # 这里收到的描述符应该是扩展后的描述符
        keypoints, scores, descriptors = self.detect_and_compute(x, self.top_k)

        pred = {
            "keypoints": keypoints + 0.5,
            "keypoint_scores": scores,
            "descriptors": descriptors,
        }
        return pred

    # 该装饰器会禁用梯度计算
    # @torch.inference_mode()
    def detect_and_compute(self, x, top_k=None):
        if top_k is None:
            top_k = self.top_k
        x, rh1, rw1 = self.preprocess_tensor(x)

        B, _, _H1, _W1 = x.shape

        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)

        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=0.05, kernel_size=2 * self.conf.nms_radius + 1)

        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :top_k]

        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)
        feats = F.normalize(feats, dim=-1)

        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)
        valid = scores > 0

        # 创建三个空列表
        keypoints_list = []
        scores_list = []
        descriptors_list = []

        # 返回值为三个独立的列表
        # keypoints = [mkpts[b][valid[b]] for b in range(B)]
        # scores = [scores[b][valid[b]] for b in range(B)]
        # descriptors = [feats[b][valid[b]] for b in range(B)]
        for b in range(B):
            valid_keypoints = mkpts[b][valid[b]]
            valid_scores = scores[b][valid[b]]
            valid_descriptors = feats[b][valid[b]]

            keypoints_list.append(valid_keypoints)
            scores_list.append(valid_scores)
            descriptors_list.append(valid_descriptors)

        # 将它们转为 pytorch 张量
        keypoints = pad_and_stack(keypoints_list, top_k, -2, mode="zeros")
        scores = pad_and_stack(scores_list, top_k, -1, mode="zeros")
        descriptors = pad_and_stack(descriptors_list, top_k, -2, mode="zeros")
        # 这里填充后每个描述子的形状为 [batch_size, num_keypoints, descriptor_dim]
        # 是否需要将描述子形状转为 [batch_size, descriptor_dim， num_keypoints]？
        # descriptors = descriptors.permute(0, 2, 1)

        '''
        为了与 lightglue 相匹配，这里需要扩展描述符的维度
        xfeat 的描述符维度是 64 维
        lightglue 的描述符维度是 256 维
        '''
        # 1. 获取描述符形状
        batch_size, num_keypoints, descriptor_dim = descriptors.shape
        # print("batch_size is:",batch_size)
        # print("num_keypoints is:",num_keypoints)
        # print("descriptor_dim is:",descriptor_dim)
        # 2. 重塑描述符
        # 将描述符从 [batch_size, descriptor_dim, num_keypoints] 转换为 [batch_size * num_keypoints, 64]
        descriptors = descriptors.view(batch_size * num_keypoints, 64)
        # 3. 通过线性层扩展描述符的维度
        # 使用线性层将描述符维度从 64 扩展到 256，并通过 ReLU 激活函数添加非线性变换
        expanded_descriptors = F.relu(self.fc(descriptors))
        # 4. 重塑扩展后的描述符
        # 将扩展后的描述符重塑回 [batch_size, num_keypoints, 256]
        expanded_descriptors = expanded_descriptors.view(batch_size, num_keypoints, 256)

        return keypoints, scores, expanded_descriptors

    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        # 将输入图像转换为 PyTorch 的张量，并将通道顺序从 HWC 转换为 CHW
        # 注意这里是 self.device
        if isinstance(x, np.ndarray) and x.shape == 3:
            x = torch.tensor(x).permute(2,0,1)[None]
        x = x.to(self.device).float()

        # 将输入图像的高度和宽度调整为可被 32 整除的最接近的值
        # 以避免深度学习模型中可能出现的锯齿状伪影
        H, W = x.shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W

        # 使用双线性插值的方式将图像的尺寸调整为经过步骤 2 调整后的高度和宽度
        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap

    def NMS(self, x, threshold=0.05, kernel_size=5):
        B, _, H, W = x.shape
        pad=kernel_size//2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        # Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos
    
    def loss(self, pred, data):
        raise NotImplementedError