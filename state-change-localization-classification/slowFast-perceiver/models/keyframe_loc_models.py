"""
Video model for Ego4D benchmark on Keyframe Localisation
"""

from math import pi, log

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import wraps
from torch import einsum

from . import head_helper
from .build import MODEL_REGISTRY
from .video_model_builder import ResNet, _POOL1


# from models.rotary import SinusoidalEmbeddings, apply_rotary_emb

# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def fourier_encode(x, max_freq, num_bands=4, base=2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(0., log(max_freq / 2) / log(base), num_bands, base=base, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# # main class
# @MODEL_REGISTRY.register()
# class Perceiver(nn.Module):
#     def __init__(
#         self, cfg
#     ):
#         super().__init__()
#         max_freq = cfg.MODEL.PERCEIVER_MAX_FREQ
#         num_freq_bands = cfg.MODEL.PERCEIVER_NUM_FREQ_BANDS
#         depth = cfg.MODEL.PERCEIVER_DEPTH
#         freq_base = cfg.MODEL.PERCEIVER_FREQ_BASE
#         input_channels = cfg.MODEL.PERCEIVER_INPUT_CHANNELS
#         input_axis = cfg.MODEL.PERCEIVER_INPUT_AXIS
#         num_latents = cfg.MODEL.PERCEIVER_NUM_LATENTS
#         latent_dim = cfg.MODEL.PERCEIVER_LATENT_DIM
#         cross_heads = cfg.MODEL.PERCEIVER_CROSS_HEADS
#         latent_heads = cfg.MODEL.PERCEIVER_LATENT_HEADS
#         cross_dim_head = cfg.MODEL.PERCEIVER_CROSS_DIM_HEAD
#         latent_dim_head = cfg.MODEL.PERCEIVER_LATENT_DIM_HEAD
#         num_classes = cfg.MODEL.PERCEIVER_NUM_CLASSES
#         attn_dropout = cfg.MODEL.PERCEIVER_ATTN_DROPOUT
#         ff_dropout = cfg.MODEL.PERCEIVER_FF_DROPOUT
#         weight_tie_layers = cfg.MODEL.PERCEIVER_WEIGHT_TIE_LAYERS
#         fourier_encode_data = cfg.MODEL.PERCEIVER_FOURIER_ENCODE_DATA
#         self_per_cross_attn = cfg.MODEL.PERCEIVER_SELF_PER_CROSS_ATTN
#         self_attn_rel_pos = cfg.MODEL.PERCEIVER_SELF_ATTN_REL_POS
#
#         self.input_axis = input_axis
#         self.max_freq = max_freq
#         self.num_freq_bands = num_freq_bands
#         self.freq_base = freq_base
#
#         input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels
#
#         self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
#
#         get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
#         get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
#         get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
#         get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
#
#         get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))
#
#         self.layers = nn.ModuleList([])
#         for i in range(depth):
#             should_cache = i > 0 and weight_tie_layers # share weights
#             cache_args = {'_cache': should_cache}
#
#             self_attns = nn.ModuleList([])
#
#             for _ in range(self_per_cross_attn):
#                 self_attns.append(nn.ModuleList([
#                     get_latent_attn(**cache_args),
#                     get_latent_ff(**cache_args)
#                 ]))
#
#             self.layers.append(nn.ModuleList([
#                 get_cross_attn(**cache_args),
#                 get_cross_ff(**cache_args),
#                 self_attns
#             ]))
#
#         # self.logit_norm = nn.LayerNorm(latent_dim)
#
#         self.to_logits = nn.Sequential(
#             nn.LayerNorm(latent_dim),
#             nn.Linear(latent_dim, num_classes)
#         )
#
#         self.to_sc_output = nn.Sequential(
#             nn.LayerNorm(latent_dim),
#             nn.Linear(latent_dim, 2)
#         )
#
#         # self.to_logits = nn.Sequential(
#         #     nn.LayerNorm(latent_dim),
#         #     nn.Linear(latent_dim, num_classes)
#         # )
#
#     # normal forward
#     def forward(self, data, mask = None):
#         if type(data) == list:
#             data = data[0]
#
#         data = data.permute(0,3,4,2,1) # batch, H, W, T, C
#         b, *axis, _, device = *data.shape, data.device
#         assert len(axis) == self.input_axis, 'input data must have the right number of axis'
#
#         # calculate fourier encoded positions in the range of [-1, 1], for all axis
#
#         axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
#         pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
#         enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base = self.freq_base)
#         enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
#         enc_pos = repeat(enc_pos, '... -> b ...', b = b)
#
#         # concat to channels of data and flatten axis
#
#         data = torch.cat((data, enc_pos), dim = -1)
#         data = rearrange(data, 'b ... d -> b (...) d')
#
#         x = repeat(self.latents, 'n d -> b n d', b = b)
#
#         for cross_attn, cross_ff, self_attns in self.layers:
#             x = cross_attn(x, context = data, mask = mask) + x
#             x = cross_ff(x) + x
#             for self_attn, self_ff in self_attns:
#                 x = self_attn(x) + x
#                 x = self_ff(x) + x
#         # average along num_latents latent arrays
#         x = x.mean(dim = -2) # batch_size  x latent_dim
#         # x = self.logit_norm(x)
#         #
#         # # return self.to_logits(x)
#         # return x
#         x, y = self.to_logits(x).unsqueeze(2), self.to_sc_output(x).unsqueeze(2)
#         return x, y


@MODEL_REGISTRY.register()
class KeyframeLocalisationClassification(ResNet):
    def _construct_network(self, cfg):
        super()._construct_network(cfg, with_head=False)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = head_helper.ResNetKeyframeLocalizationHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES[0],
            pool_size=[
                [
                    1,
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ]
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        self.head_name = "head_kf_loc_class"
        self.add_module(self.head_name, head)


@MODEL_REGISTRY.register()
class KeyframeLocalisationRegression(ResNet):
    """
    Not a scalable approach but, adding code here for the sake of completeness
    """

    def _construct_network(self, cfg):
        super()._construct_network(cfg, with_head=False)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = head_helper.ResNetRegressionHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES[0],
            pool_size=[
                [
                    1,
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ]
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT
        )
        self.head_name = "head_kf_loc_regress"
        self.add_module(self.head_name, head)
