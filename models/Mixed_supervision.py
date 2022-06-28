__author__="Hao Bian"
import math
import numpy as np
import torch
import torch.nn as nn
from timm.models import create_model
from timm.models.layers import trunc_normal_
from .builder import MODELS

from utils.util import read_yaml

from .layers import *


def get_block(block_type, **kargs):
    if block_type == 'mha':
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type == 'ffn':
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type == 'tr':
        # transformer block
        return Block(**kargs)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_dpr(drop_path_rate, depth, drop_path_decay='linear'):
    if drop_path_decay == 'linear':
        # linear dpr decay
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    elif drop_path_decay == 'fix':
        # use fixed dpr
        dpr = [drop_path_rate] * depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate) == depth
        dpr = drop_path_rate
    return dpr


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)  # N, D
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :,
        :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class FixedPositionalEncoding_2d(nn.Module):
    def __init__(self, embedding_dim, height, width):
        super(FixedPositionalEncoding_2d, self).__init__()
        pe = positionalencoding2d(embedding_dim, height, width)  # 编码一个最长的长度
        self.register_buffer('pe', pe)

    def forward(self, x, coord):

        pos = torch.stack([torch.stack([self.pe[:, x, y] for (x, y) in batch])
                          for batch in (coord / 100).long()])

        x = x + 0.1 * pos
        return x


@MODELS.register_module()
class Mixed_supervision(nn.Module):
    """ Mixed_supervision with Vision Transformer
    Arguements:
        masking_ratio: The masking ratio of random masking strategy (default: 0.5)
        num_classes: The slide-level class nummbers (default: 4)
        embed_dim: The instance feature dimension (default: 1280)
        depth: The numbers of Transformer blocks (default: 2)
        num_heads: The numbers of Transformer block head (default: 12)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """

    def __init__(self, masking_ratio=0.5, num_classes=4, embed_dim=1280, depth=2,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', norm_layer=nn.LayerNorm, head_dim=None,
                 skip_lam=1.0, order=None):
        super().__init__()
        self.masking_ratio = masking_ratio
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.output_dim = embed_dim if num_classes == 0 else num_classes

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.position_encoding_xy = FixedPositionalEncoding_2d(
            self.embed_dim, 200, 200)

        if order is None:
            # [0.0, 0.0066666668, 0.0133333336, ..., 0.1]
            dpr = get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(depth)])
        else:
            # use given order to sequentially generate modules
            dpr = get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList([
                get_block(order[i],
                          dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(len(order))])

        self.norm = norm_layer(embed_dim)
        self.slide_head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # token head
        self.instance_head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # init weight
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.slide_head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.slide_head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_tokens(self, x, coords):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if coords is not None:
            x = self.position_encoding_xy(x, coords)
        # [B, num_patches + 1， embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def random_masking(self, x, coords):
        N = len(coords[0])
        S = int(N * (1 - self.masking_ratio))
        if S == 1:  # Avoid situations where there is only one instance selected
            S = 2
        index = torch.LongTensor(np.random.choice(
            range(N), S, replace=False)).to(x.device)
        x = torch.index_select(x, 1, index)
        coords = torch.index_select(coords, 1, index)
        return x, coords, index

    def forward(self, x, coords=None, phase='test'):

        # random masking strategy
        if phase != 'test':
            x, coords, index = self.random_masking(x, coords)

        # token interaction
        x = self.forward_tokens(x, coords)  # [B, num_patches + 1, embed_dim]

        # slide-level prediction
        x_cls = self.slide_head(x[:, 0])  # [B, num_classes]

        #
        x_aux = self.instance_head(x[:, 1:])  # [B, num_patches, num_classes]

        if phase == 'test':
            return x_cls, x_aux
        else:
            return x_cls, x_aux, index
