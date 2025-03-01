# Copyright (c) 2023, Tri Dao, Albert Gu.
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from .selective_scan_interface import SMSS_fn,CMSS_fn
class S6(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        dt_rank=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_model,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_model, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_model, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_model, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_model,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_model, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
    def forward(self, modality1):
        """
        modality1: (B, L, 2*d_model)
        Returns: same shape as modality1
        """
        A = -torch.exp(self.A_log.float())  # (d_model, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        out = SMSS_fn(
            modality1,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return rearrange(out, "b l d -> b d l")
        
        
        
class CMS6(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        dt_rank=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.conv1d_y = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_model,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_model,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_model, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_model, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_model, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_model,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_model, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
    def forward(self, modality1, modality2):
        """
        modality1: (B, L, d_model)
        modality2: (B, L, 2*d_model)
        Returns: same shape as modality1
        """
        A = -torch.exp(self.A_log.float())  # (d_model, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        out = CMSS_fn(
            modality2,
            modality1,
            self.conv1d_y.weight,
            self.conv1d_y.bias,
            self.conv1d_x.weight,
            self.conv1d_x.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return rearrange(out, "b l d -> b d l")        