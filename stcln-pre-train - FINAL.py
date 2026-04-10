# Generated from: stcln-pre-train - FINAL.ipynb
# Converted at: 2026-04-10T15:50:36.811Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # 🛰️ STCLN Pre-training on PASTIS-R
# 
# **Spatio-Temporal Contrastive Learning Network — Masked Time-Step Imputation Pre-training**
# 
# ---
# 
# ### Overview
# This notebook pre-trains the **STCLN (UTAE backbone)** on the [PASTIS-R dataset](https://github.com/VSainteuf/pastis-benchmark) using a **masked spectral reconstruction** proxy task:
# 
# - Randomly mask temporal observations (dropout-style, p=0.4)
# - NDVI-aware masking: vegetation patches are preferentially masked
# - The model learns to reconstruct the original pixel values from unmasked context
# - Loss: MSE computed **only on masked positions**
# 
# ### Architecture
# ```
# Input (B, T, C, H, W)
#     → DoubleConv  → 32 channels
#     → DoubleConv  → 256 channels   (no spatial downsampling)
#     → LTAE2d      → Transformer (3 layers, 8 heads, d=256)
#     → Linear head → reconstruct C spectral bands
# ```
# 
# ### Dataset
# - **PASTIS-R**: Sentinel-2 multi-spectral time series over French agricultural parcels
# - Using **fold 5** for pre-training (same as original paper)
# - Spatial tiling: each 128×128 patch split into 16 sub-patches of 32×32 per iteration
# 
# ---


# ## 1. Environment Setup


import subprocess, sys

# Install required packages not available by default on Kaggle
subprocess.run([sys.executable, '-m', 'pip', 'install', 'geopandas', '--quiet'], check=True)

print('✅ Environment ready')

import os
import sys
import copy
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer as TorchTEL
from torch.nn.modules import LayerNorm
class SummaryWriter:
    """Minimal stub — avoids TensorFlow/TensorBoard conflict on Kaggle."""
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, 'loss_log.txt')
    def add_scalar(self, tag, value, step):
        with open(self.log_path, 'a') as f:
            f.write(f'{step},{tag},{value:.6f}\n')
    def close(self):
        pass

print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU             : {torch.cuda.get_device_name(0)}')
    print(f'VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ## 2. Configuration
# 
# All hyperparameters and paths are centralised here. Edit this cell before running.


# ── Paths ──────────────────────────────────────────────────────────────────
DATADIR       = '/kaggle/input/datasets/mamainwuxi/pastis-r/PASTIS-R'
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
RUNS_DIR      = '/kaggle/working/runs'

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE    = 4       # P100 16 GB — increase to 4 if memory allows
WORKERS       = 4
EPOCHS        = 100
LR            = 1e-4
SNAPSHOT      = None    # Path to a .tar checkpoint to resume from, or None

# ── Model (do not change — must match V1/V2 architecture) ──────────────────
N_CHANNELS    = 10
N_CLASSES     = 20
ENC_WIDTHS    = [32, 256]
DEC_WIDTHS    = [32, 256]
AGG_MODE      = 'att_mean'
N_HEAD        = 8
D_MODEL       = 256
D_K           = 32
DROPOUT       = 0.4
NUM_FEATURES  = 10

# ── Dataset ────────────────────────────────────────────────────────────────
FOLDS         = [5]     # Same fold used in both V1 and V2

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')
print(f'Config ready ✅')

# ## 3. PASTIS-R Dataset Loader
# 
# Minimal self-contained loader for Kaggle — no external `src/` package required.
# Reads directly from the PASTIS-R folder structure.


import numpy as _np   # ← ADD THIS AS VERY FIRST LINE
import geopandas as gpd
import pandas as pd

class PASTIS_Dataset(torch.utils.data.Dataset):

    def __init__(self, folder, norm=True, target='semantic', folds=None,
                 reference_date='2018-09-01', class_mapping=None):
        super().__init__()
        self.folder         = folder
        self.norm           = norm
        self.target         = target
        self.reference_date = pd.Timestamp(reference_date)

        meta_path  = os.path.join(folder, 'metadata.geojson')
        self.meta  = gpd.read_file(meta_path)

        if folds is not None:
            self.meta = self.meta[self.meta['Fold'].isin(folds)].reset_index(drop=True)

        self.len = len(self.meta)

        if norm:
            norm_path = os.path.join(folder, 'NORM_S2_patch.json')
            with open(norm_path, 'r') as f:
                norm_vals = json.load(f)
            all_means = np.array([norm_vals[k]['mean'] for k in norm_vals])
            all_stds  = np.array([norm_vals[k]['std']  for k in norm_vals])
            self.norm_mean = torch.tensor(all_means.mean(axis=0), dtype=torch.float32)
            self.norm_std  = torch.tensor(all_stds.mean(axis=0),  dtype=torch.float32)

        print(f'PASTIS_Dataset: {self.len} patches (folds={folds})')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        row      = self.meta.iloc[idx]
        patch_id = row['ID_PATCH']

        # ── Load S2 data ───────────────────────────────────────────────────
        s2_path = os.path.join(self.folder, 'DATA_S2', f'S2_{patch_id}.npy')  # ← THIS LINE WAS MISSING
        data = torch.from_numpy(_np.load(s2_path).astype(_np.float32))
        # data is (T, C, H, W) confirmed from diagnostic
        data = data[:, :N_CHANNELS, :, :]  # (T, 10, H, W)

        if self.norm:
            mean = self.norm_mean[:N_CHANNELS].view(1, N_CHANNELS, 1, 1)
            std  = self.norm_std[:N_CHANNELS].view(1, N_CHANNELS, 1, 1).clamp(min=1e-6)
            data = (data - mean) / std

        # ── Dates ──────────────────────────────────────────────────────────
        if 'dates-S2' in row:
            date_dict = row['dates-S2']
            date_list = [date_dict[k] for k in sorted(date_dict.keys(), key=lambda x: int(x))]
            dates = []
            for d in date_list:
                d_str = str(int(d))
                ts    = pd.Timestamp(year=int(d_str[0:4]),
                                     month=int(d_str[4:6]),
                                     day=int(d_str[6:8]))
                dates.append((ts - self.reference_date).days)
            dates = torch.tensor(dates, dtype=torch.long)
        else:
            dates = torch.arange(data.shape[0], dtype=torch.long)

        # ── Label ──────────────────────────────────────────────────────────
        label_path = os.path.join(self.folder, 'ANNOTATIONS', f'TARGET_{patch_id}.npy')
        label = torch.from_numpy(_np.load(label_path).astype(_np.int64))
        if label.ndim == 3:
            label = label[0]

        return (data, dates), label


def pad_collate(batch, pad_value=0):
    # batch is a list of: ((data, dates), label)
    # Unpack manually instead of relying on zip(*batch)
    datas  = [item[0][0] for item in batch]   # list of (T, C, H, W)
    datess = [item[0][1] for item in batch]   # list of (T,)
    labels = [item[1]    for item in batch]   # list of (H, W)

    max_t = max(d.shape[0] for d in datas)

    padded_data  = []
    padded_dates = []

    for d, dt in zip(datas, datess):
        pad_len = max_t - d.shape[0]
        if pad_len > 0:
            pad_d  = torch.full((pad_len, d.shape[1], d.shape[2], d.shape[3]),
                                pad_value, dtype=d.dtype)
            d  = torch.cat([d,  pad_d], dim=0)
            dt = torch.cat([dt, torch.zeros(pad_len, dtype=dt.dtype)], dim=0)
        padded_data.append(d)
        padded_dates.append(dt)

    return (
        torch.stack(padded_data,  dim=0),   # (B, T, C, H, W)
        torch.stack(padded_dates, dim=0)    # (B, T)
    ), torch.stack(labels, dim=0)           # (B, H, W)


print('Dataset classes defined ✅')

# ## 4. Model Definition (STCLN — exact copy of V1/V2)
# 
# > ⚠️ **No architectural changes made.** This is a verbatim copy of `STCLN.py` from both versions.


# ═══════════════════════════════════════════════════════════════════════════
#  STCLN.py  —  Spatio-Temporal Contrastive Learning Network
#  Exact architecture from V1/V2. Zero architectural changes.
# ═══════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),   # intentionally disabled — no spatial downsampling
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


# ── Positional Encoders ─────────────────────────────────────────────────────

class PositionalEncoder(nn.Module):
    def __init__(self, d, T=30, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = batch_positions[:, :, None] / self.denom[None, None, :]
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])
        if self.repeat is not None:
            sinusoid_table = torch.cat([sinusoid_table for _ in range(self.repeat)], dim=-1)
        return sinusoid_table


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=366):
        super().__init__()
        pe = torch.zeros(max_len + 1, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[1:, 0::2] = torch.sin(position * div_term)
        pe[1:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, time):
        output = torch.stack([torch.index_select(self.pe, 0, time[i, :]) for i in range(time.shape[0])], dim=0)
        return output


# ── Transformer Encoder Layer (custom, with attn return) ───────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.dropout   = nn.Dropout(dropout)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, weights = self.self_attn(src, src, src,
                                       attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights  # weights -> (N, l, l)


# ── Temporal Aggregator ────────────────────────────────────────────────────

class Temporal_Aggregator(nn.Module):
    def __init__(self, mode='mean'):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == 'att_group':
                n_heads, b, t, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t * t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:], mode='bilinear', align_corners=False)(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, t, *x.shape[-2:])
                out  = torch.stack(x.chunk(n_heads, dim=2))
                out  = torch.matmul(attn.permute(0,1,4,5,2,3),
                                    out.permute(0,1,4,5,2,3)).permute(0,1,5,4,2,3)
                out  = torch.cat([group for group in out], dim=1)
                return out
            elif self.mode == 'att_mean':
                attn = attn_mask.mean(dim=0)
                attn = nn.Upsample(size=x.shape[-2:], mode='bilinear', align_corners=False)(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out  = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == 'mean':
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == 'att_group':
                n_heads, b, t, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t * t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:], mode='bilinear', align_corners=True)(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, t, *x.shape[-2:])
                out  = torch.stack(x.chunk(n_heads, dim=2))
                out  = torch.matmul(attn.permute(0,1,4,5,2,3),
                                    out.permute(0,1,4,5,2,3)).permute(0,1,4,5,2,3)
                out  = torch.cat([group for group in out], dim=2)
                return out
            elif self.mode == 'att_mean':
                n_heads, b, t, t, h, w = attn_mask.shape
                attn = attn_mask.mean(dim=0)
                attn = attn.view(b, t * t, h, w)
                attn = nn.Upsample(size=x.shape[-2:], mode='bilinear', align_corners=True)(attn)
                attn = attn.view(b, t, t, *attn.shape[-2:])
                out  = torch.matmul(attn.permute(0,3,4,1,2),
                                    x.permute(0,3,4,1,2)).permute(0,3,4,1,2)
                return out
            elif self.mode == 'mean':
                return x.permute(0, 2, 1, 3, 4)


# ── LTAE2d ─────────────────────────────────────────────────────────────────

class LTAE2d(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=4, mlp=[256, 128],
                 dropout=0.1, d_model=256, T=1000, return_att=False,
                 positional_encoding=True):
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp         = copy.deepcopy(mlp)
        self.return_att  = return_att
        self.n_head      = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv  = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv  = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        # 3-layer Transformer stack (used in forward; same as V1/V2)
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model * 4, dropout=0.1)
            for _ in range(3)
        ])

        self.in_norm  = nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend([
                nn.Linear(self.mlp[i], self.mlp[i + 1]),
                nn.BatchNorm1d(self.mlp[i + 1]),
                nn.ReLU(),
            ])
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        # kept for weight compatibility with saved checkpoints
        encoder_layer = TransformerEncoderLayer(256, 8, 256 * 4, 0.1)
        encoder_norm  = LayerNorm(256)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 3, encoder_norm)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape

        if pad_mask is not None:
            pad_mask = (pad_mask.unsqueeze(-1).repeat((1, 1, h))
                                .unsqueeze(-1).repeat((1, 1, 1, w)))
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (batch_positions.unsqueeze(-1).repeat((1, 1, h))
                                 .unsqueeze(-1).repeat((1, 1, 1, w)))
            bp  = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        for enc_layer in self.layer_stack:
            out, attn = enc_layer(out)

        attn = attn.view(sz_b, h, w, seq_len, seq_len).unsqueeze(0).permute(0, 1, 4, 5, 2, 3)

        if self.return_att:
            return out, attn
        else:
            return out


# ── UTAE Backbone ───────────────────────────────────────────────────────────

class UTAE(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,
                 encoder_widths=[32, 64, 256],
                 decoder_widths=[64, 64, 256],
                 agg_mode='att_group',
                 n_head=8, d_model=256, d_k=32):
        super(UTAE, self).__init__()
        self.n_channels      = n_channels
        self.n_classes       = n_classes
        self.d_model         = d_model
        self.bilinear        = bilinear
        self.encoder_widths  = encoder_widths
        self.decoder_widths  = decoder_widths

        self.inc   = DoubleConv(n_channels, encoder_widths[0])
        self.down1 = Down(encoder_widths[0], encoder_widths[1])
        self.up3   = Up(decoder_widths[-1] + encoder_widths[-2], decoder_widths[-2], bilinear)

        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

    def forward(self, x, batch_positions=None):
        b_x = x.shape[0]
        l_x = x.shape[1]
        d_x = x.shape[2]
        h_x = x.shape[-1]

        x   = x.view(-1, *x.shape[-3:])       # (b*l, d, h, w)
        x1  = self.inc(x)                      # (b*l, 32, h, w)
        x2  = self.down1(x1)                   # (b*l, 256, h, w) — no spatial downsample

        obs_embed = x2.view(b_x, l_x, -1, h_x, h_x)   # (b, l, 256, h, w)
        obs_embed, attn = self.temporal_encoder(obs_embed, batch_positions=batch_positions)

        obs_embed = (obs_embed
                     .view(b_x, h_x, h_x, l_x, -1)
                     .permute(0, 3, 4, 1, 2)
                     .contiguous()
                     .view(b_x * l_x, -1, h_x, h_x))   # (b*l, 256, h, w)

        obs_embed = (obs_embed
                     .view(b_x, l_x, -1, h_x, h_x)
                     .permute(0, 3, 4, 1, 2))           # (b, h, w, l, 256)

        return obs_embed, x2


# ── Pre-training Head ───────────────────────────────────────────────────────

class UTAEPrediction(nn.Module):
    """
    Proxy task: missing-data imputation.
    Randomly masks temporal observations and reconstructs the original pixels.
    """
    def __init__(self, utae: UTAE, num_features=13, dropout=0.7):
        super().__init__()
        self.utae       = utae
        self.linear     = nn.Linear(self.utae.encoder_widths[1], num_features)
        self.midlinear  = nn.Linear(self.utae.encoder_widths[1], num_features)
        self.dropout    = dropout
        self.MASK_TOKEN = nn.Parameter(torch.zeros(1))

    def forward(self, x, cluster_id_x, pos):
        target = x.clone()

        # ── Spatial-temporal dropout mask ─────────────────────────────────
        # Correct: mask individual pixels across spatial+temporal dimensions
       # CORRECT — V1 exact temporal masking
        mask = torch.ones(x.shape[0], x.shape[1], x.shape[3], x.shape[4]).cuda()
        mask = F.dropout(mask, self.dropout) * (1 - self.dropout)
        mask = mask.unsqueeze(2).repeat((1, 1, x.shape[2], 1, 1))  # (b,l,c,h,w)

        # ── NDVI-aware override: always keep non-vegetation visible ────────
        clusterLmean = torch.mean(cluster_id_x.float(), dim=[3, 4], keepdim=True).repeat(
            (1, 1, 1, mask.shape[3], mask.shape[4])
        )
        mask[clusterLmean <= 0.9] = 1

        x = x.clone()
        x[mask == 0] = self.MASK_TOKEN

        x, x2 = self.utae(x, pos)  # (b, h, w, l, 256)

        out_main = self.linear(x).permute(0, 3, 4, 1, 2)  # (b, l, c, h, w)
        out_mid  = (self.midlinear(x2.permute(0, 2, 3, 1))
                        .permute(0, 3, 1, 2)
                        .view(x.shape[0], x.shape[3], -1, *x2.shape[-2:]))  # (b, l, c, h, w)

        return out_main, out_mid, target, mask


# ── Fine-tuning Head (included for completeness / checkpoint compatibility) ─

class UTAEClassification(nn.Module):
    """Downstream task: Satellite Time Series Classification"""
    def __init__(self, utae: UTAE):
        super().__init__()
        self.utae      = utae
        self.outlinear = nn.Linear(self.utae.encoder_widths[-1], self.utae.n_classes)
        self.conv1     = nn.Conv2d(self.utae.encoder_widths[-1], self.utae.encoder_widths[-1], (1,1), bias=False)
        self.conv2     = nn.Conv2d(self.utae.encoder_widths[-1], self.utae.encoder_widths[-1], (1,1), bias=False)
        self.conv3     = nn.Conv2d(self.utae.encoder_widths[-1], self.utae.encoder_widths[-1], (1,1), bias=False)
        self.l         = nn.Parameter(torch.zeros(1))
        self.dropout   = nn.Dropout(0.1)

    def forward(self, x, pos):
        out, x2   = self.utae(x, pos)
        out_1     = self.conv1(
            out.permute(0,3,4,1,2).view(-1, out.shape[-1], out.shape[1], out.shape[1])
        ).view(out.shape[0], -1, out.shape[1]**2).permute(0, 2, 1)
        out, _    = torch.max(out, dim=3)
        x2_2      = self.conv2(x2).view(out.shape[0], -1, x2.shape[-1]**2)
        attn      = torch.matmul(out_1, x2_2) / np.power(self.utae.encoder_widths[-1], 0.5)
        attn      = nn.Softmax(dim=2)(attn)
        out       = self.conv3(out.permute(0,3,1,2)).permute(0,2,3,1)
        out       = (self.l * torch.bmm(attn, out.view(-1, out.shape[1]**2, self.utae.encoder_widths[-1]))
                     + out.view(-1, out.shape[1]**2, self.utae.encoder_widths[-1]))
        out       = self.outlinear(out).permute(0,2,1).view(out.shape[0], -1, x2.shape[-1], x2.shape[-1])
        return out, x2


print('Model classes defined ✅')

# ## 5. Initialise Dataset, Model & Optimizer


# ── Dataset & DataLoader ────────────────────────────────────────────────────
dataset = PASTIS_Dataset(
    folder=DATADIR,
    norm=True,
    target='semantic',
    folds=FOLDS
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,       # ← change from 2 to 0
    pin_memory=True,    # ← change from True to False
    collate_fn=pad_collate
)

print(f'Batches per epoch : {len(dataloader)}')
print(f'Gradient steps/epoch (×16 tiles) : {len(dataloader) * 16}')

# ── Model ───────────────────────────────────────────────────────────────────
model = UTAEPrediction(
    UTAE(
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        bilinear=True,
        encoder_widths=ENC_WIDTHS,
        decoder_widths=DEC_WIDTHS,
        agg_mode=AGG_MODE,
        n_head=N_HEAD,
        d_model=D_MODEL,
        d_k=D_K
    ),
    num_features=NUM_FEATURES,
    dropout=DROPOUT
)

# ── Optimizer & Loss ────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = nn.MSELoss(reduction='none')
from torch.cuda.amp import GradScaler
scaler = GradScaler()

# ── Move to GPU ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    loss_fn = loss_fn.cuda()

# ── Count parameters ────────────────────────────────────────────────────────
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters : {n_params:,}')

# ── Resume from snapshot if provided ────────────────────────────────────────
start_epoch = 0
if SNAPSHOT is not None:
    checkpoint  = torch.load(SNAPSHOT)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Move optimizer states to GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    print(f'Resumed from epoch {start_epoch}')

writer = SummaryWriter(RUNS_DIR)
print('Model ready ✅')

# ## 6. Pre-training Loop
# 
# Each epoch iterates over the dataloader. Every batch is additionally split into **16 spatial tiles (4×4)** — exactly as in V1 — giving more gradient steps per sample and acting as a form of data augmentation.
# 
# **Checkpoints are saved every epoch** to `/kaggle/working/checkpoints/`.  
# The **UTAE backbone weights** are also exported separately after each epoch for direct use in fine-tuning.


def save_checkpoint(epoch, model, optimizer, checkpoint_dir):
    """Save full checkpoint + separate UTAE backbone weights (for fine-tuning)."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Full checkpoint (resume training)
    full_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        ),
        'optimizer_state_dict': optimizer.state_dict(),
    }, full_path)

    # Backbone only (for fine-tuning downstream)
    utae_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.utae.tar')
    utae_model = model.module.utae if isinstance(model, nn.DataParallel) else model.utae
    torch.save(utae_model.state_dict(), utae_path)

    print(f'  └─ Saved: {full_path}')
    print(f'  └─ Saved: {utae_path}')


print('save_checkpoint defined ✅')

import time

print('='*60)
print('  STCLN Pre-training — PASTIS-R')
print(f'  Epochs      : {start_epoch} → {EPOCHS}')
print(f'  Batch size  : {BATCH_SIZE}')
print(f'  Device      : {DEVICE}')
print('='*60)

for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss  = 0.0
    n_steps     = 0
    t_start     = time.time()

    for iteration, data in enumerate(dataloader):
        (inp, dates), label = data
        dates = dates.cuda()
        inp = inp.cuda()  # (B, T, C, H, W)

        # ── 4×4 spatial tiling (same as V1 / V2) ─────────────────────────
        for i in range(4):
            for j in range(4):
                optimizer.zero_grad()

                split     = inp.shape[-1] // 4
                inp_patch = inp[:, :, :,
                                i * split:(i + 1) * split,
                                j * split:(j + 1) * split]  # (B, T, C, split, split)

                # ── NDVI-based cluster mask ───────────────────────────────
                # Band indices: 2=Red, 6=NIR  (0-indexed S2 bands)
                ndvi = (inp_patch[:, :, 6, :, :] - inp_patch[:, :, 2, :, :]) / (
                        inp_patch[:, :, 2, :, :] + inp_patch[:, :, 6, :, :] + 1e-20
                )
                cluster_ids_x = (
                    ndvi.gt(0.2)
                       .int()
                       .unsqueeze(2)
                       .repeat(1, 1, inp_patch.shape[2], 1, 1)
                )  # (B, T, C, split, split)

                # ── Integer positional indices (same as V1 / V2) ──────────
                # ── Real acquisition dates (day offsets from reference date) ──
                pos = dates[:, :inp_patch.shape[1]].cuda()

                # ── Forward + loss ─────────────────────────────────────────
                with torch.cuda.amp.autocast():
                    output, _, target, mask = model(inp_patch, cluster_ids_x, pos)
                    l = loss_fn(output, target.float())
                    l = (l * (1 - mask).float()).sum() / ((1 - mask).sum() + 1e-20)
                
                scaler.scale(l).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += l.item()
                n_steps    += 1

        # ── Per-batch progress print ──────────────────────────────────────
        if (iteration + 1) % 20 == 0 or (iteration + 1) == len(dataloader):
            avg = epoch_loss / n_steps
            elapsed = time.time() - t_start
            print(f'  Epoch {epoch:>3d} | Iter {iteration+1:>4d}/{len(dataloader)} '
                  f'| AvgLoss {avg:.6f} | Elapsed {elapsed:.0f}s')

    # ── End of epoch ─────────────────────────────────────────────────────────
    avg_epoch_loss = epoch_loss / n_steps
    writer.add_scalar('Loss/train', avg_epoch_loss, epoch)

    print(f'\nEpoch {epoch} complete — AvgLoss: {avg_epoch_loss:.6f}')

    save_checkpoint(epoch, model, optimizer, CHECKPOINT_DIR)

writer.close()
print('\n✅ Pre-training complete!')
print(f'Checkpoints saved to: {CHECKPOINT_DIR}')

# ## 7. Verify Saved Files


saved = sorted(os.listdir(CHECKPOINT_DIR))
print(f'Files in {CHECKPOINT_DIR}:')
for f in saved:
    size_mb = os.path.getsize(os.path.join(CHECKPOINT_DIR, f)) / 1e6
    print(f'  {f:50s}  {size_mb:.1f} MB')

# ## 8. Quick Sanity Check — Load & Inspect Backbone
# 
# Verifies the saved UTAE backbone weights load cleanly and have the expected parameter count.


# Load the final epoch backbone
final_epoch = EPOCHS - 1
utae_path   = os.path.join(CHECKPOINT_DIR, f'checkpoint_{final_epoch}.utae.tar')

backbone = UTAE(
    n_channels=N_CHANNELS,
    n_classes=N_CLASSES,
    bilinear=True,
    encoder_widths=ENC_WIDTHS,
    decoder_widths=DEC_WIDTHS,
    agg_mode=AGG_MODE,
    n_head=N_HEAD,
    d_model=D_MODEL,
    d_k=D_K
)

state = torch.load(utae_path, map_location='cpu')
backbone.load_state_dict(state)
backbone.eval()

n_params = sum(p.numel() for p in backbone.parameters())
print(f'Backbone loaded successfully ✅')
print(f'Total parameters : {n_params:,}')
print(f'Checkpoint path  : {utae_path}')

# ---
# 
# ## Summary
# 
# | Item | Value |
# |---|---|
# | Model | UTAE (STCLN backbone) |
# | Pre-training task | Masked spectral reconstruction (MSE) |
# | Dataset | PASTIS-R, fold 5, Sentinel-2 |
# | Input channels | 10 |
# | Encoder widths | [32, 256] |
# | Transformer layers | 3 × (8-head, d=256, FFN=1024) |
# | Masking dropout | 0.4 with NDVI-aware override |
# | Spatial tiling | 4×4 per batch |
# | Optimizer | Adam, lr=1e-4, grad clip=5 |
# | Saved artifacts | `checkpoint_N.tar` (full) + `checkpoint_N.utae.tar` (backbone) |
# 
# The `.utae.tar` files are ready to be loaded as a pre-trained backbone for downstream fine-tuning (crop type classification, etc.).