import timm
import torch
import torch.nn as nn
from einops import rearrange
from .tensor_ops import cus_sample
from .ssm_modules import S6,CMS6
from .saliency_module import CBAM, SEM

def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    else:
        raise NotImplementedError
class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
    ):
        super().__init__()
        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name, inplace=False))
class StackedCBRBlock(nn.Sequential):
    def __init__(self, in_c, out_c, num_blocks=1, kernel_size=3):
        assert num_blocks >= 1
        super().__init__()
        if kernel_size == 3:
            kernel_setting = dict(kernel_size=3, stride=1, padding=1)
        elif kernel_size == 1:
            kernel_setting = dict(kernel_size=1)
        else:
            raise NotImplementedError
        cs = [in_c] + [out_c] * num_blocks
        self.channel_pairs = self.slide_win_select(cs, win_size=2, win_stride=1, drop_last=True)
        self.kernel_setting = kernel_setting
        for i, (i_c, o_c) in enumerate(self.channel_pairs):
            self.add_module(name=f"cbr_{i}", module=ConvBNReLU(i_c, o_c, **self.kernel_setting))
    @staticmethod
    def slide_win_select(items, win_size=1, win_stride=1, drop_last=False):
        num_items = len(items)
        i = 0
        while i + win_size <= num_items:
            yield items[i : i + win_size]
            i += win_stride
        if not drop_last:
            yield items[i : i + win_size]
class ConvFFN(nn.Module):
    def __init__(self, dim, out_dim=None, ffn_expand=4):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            StackedCBRBlock(dim, dim * ffn_expand, num_blocks=1, kernel_size=3),
            nn.Conv2d(dim * ffn_expand, out_dim, 1),
        )
    def forward(self, x):
        return self.net(x)
       
class SMFB(nn.Module):
    def __init__(self, dim,d_state=64):
        super().__init__()
        self.to_x = nn.Conv2d(dim, dim, 1)
        self.mambaF = S6(d_model=dim//2,d_state=d_state,d_conv=4)
        self.mambaR = S6(d_model=dim//2,d_state=d_state,d_conv=4)
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        N, C, H, W = x.shape
        x = self.to_x(x)
        x = rearrange(x, "b d h w -> b d (h w)")
        xF,xR = x,x #.chunk(2,dim=1)
        xR = torch.flip(xR,[-1])
        xF = self.mambaF(xF)
        xR = self.mambaR(xR)
        xR = torch.flip(xR,[-1])
        # x = xF+xR
        x = torch.cat([xF, xR], dim=1)
        x = rearrange(x, "b d (h w) -> b d h w",h=H,w=W)
        x = self.proj(x)
        return x
class CMFB(nn.Module):
    def __init__(self, dim,d_state=64):
        super().__init__()
        self.to_x = nn.Conv2d(dim, dim//2, 1)
        self.to_y = nn.Conv2d(dim, dim, 1)
        self.mambaF = CMS6(d_model=dim//2,d_state=d_state,d_conv=4)
        self.mambaR = CMS6(d_model=dim//2,d_state=d_state,d_conv=4)
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, x, y):
        N, C, H, W = x.shape
        x = self.to_x(x)
        x = rearrange(x, "b d h w -> b d (h w)")
        xF,xR = x,x #x.chunk(2,dim=1)
        xR = torch.flip(xR,[-1])
        y = self.to_y(y)
        y = rearrange(y, "b d h w -> b d (h w)")
        yF,yR = y,y #y.chunk(2,dim=1)
        yR = torch.flip(yR,[-1])
        xF = self.mambaF(xF,yF)
        xR = self.mambaR(xR,yR)
        xR = torch.flip(xR,[-1])
        # x = xF+xR 
        x = torch.cat([xF, xR], dim=1)
        x = rearrange(x, "b d (h w) -> b d h w",h=H,w=W)
        x = self.proj(x)
        return x
    
        
class SMDB(nn.Module):
    def __init__(self, dim,d_state=64, ffn_expand=1):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.smfb = SMFB(dim,d_state=d_state)
        self.cbam=CBAM(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = ConvFFN(dim=dim, ffn_expand=ffn_expand, out_dim=dim)
    def forward(self, x1):
        normed_x = self.norm1(x1)
        x = x1+self.smfb(normed_x) 
        x= self.cbam(x)
        x = x + self.ffn(self.norm2(x))
        return x
class CMDB(nn.Module):
    def __init__(self, dim,d_state=64, ffn_expand=1):
        super().__init__()
        self.rgb_norm2 = nn.BatchNorm2d(dim)
        self.depth_norm2 = nn.BatchNorm2d(dim)
        self.depth_to_rgb = CMFB(dim,d_state=d_state)
        self.rgb_to_depth = CMFB(dim,d_state=d_state)
        self.cbam1=CBAM(dim)
        self.cbam2=CBAM(dim)
        self.norm3 = nn.BatchNorm2d(2*dim)
        self.ffn = ConvFFN(dim=2*dim, ffn_expand=ffn_expand, out_dim=2*dim)
    def forward(self, rgb, depth):
        normed_rgb = self.rgb_norm2(rgb)
        normed_depth = self.depth_norm2(depth)
        transd_rgb = self.depth_to_rgb(normed_rgb, normed_depth)
        rgb_rgbd = rgb + transd_rgb
        rgb_rgbd=self.cbam1(rgb_rgbd)
        transd_depth = self.rgb_to_depth(normed_depth, normed_rgb) 
        depth_rgbd = depth + transd_depth
        depth_rgbd=self.cbam2(depth_rgbd)
        rgbd = torch.cat([rgb_rgbd, depth_rgbd], dim=1)
        rgbd = rgbd + self.ffn(self.norm3(rgbd))
        return rgbd
class M2DB(nn.Module):
    def __init__(self, in_dim, embed_dim, ffn_expand):
        super().__init__()
        self.rgb_cnn_proj = nn.Sequential(
            StackedCBRBlock(in_c=in_dim, out_c=embed_dim), nn.Conv2d(embed_dim, embed_dim, 1)
        )
        self.depth_cnn_proj = nn.Sequential(
            StackedCBRBlock(in_c=in_dim, out_c=embed_dim), nn.Conv2d(embed_dim, embed_dim, 1)
        )
        self.rgb_smdb = SMDB(embed_dim,d_state=64,ffn_expand=ffn_expand)
        self.depth_smdb = SMDB(embed_dim,d_state=64,ffn_expand=ffn_expand)
        self.cmdb = CMDB(embed_dim,d_state=64,ffn_expand=ffn_expand)
        self.smdb = SMDB(2*embed_dim,d_state=64, ffn_expand=ffn_expand)
    def forward(self, rgb, depth,previous=None):
        rgb = self.rgb_cnn_proj(rgb)
        depth = self.depth_cnn_proj(depth)
        rgb = self.rgb_smdb(rgb)
        depth = self.depth_smdb(depth)
        rgbd = self.cmdb(rgb, depth)
        if previous is not None:
            rgbd = rgbd + previous
        rgbd = self.smdb(rgbd)
        return rgbd

class SSNet(nn.Module):
    def __init__(self, embed_dim=64, pretrained=None):
        super().__init__()
        self.fem1: nn.Module = timm.create_model(
            model_name="resnet101d", features_only=True, out_indices=range(1, 5)
        )
        self.fem2: nn.Module = timm.create_model(
            model_name="resnet101d", features_only=True, out_indices=range(1, 5)
        )
        if pretrained:
            self.fem1.load_state_dict(torch.load(pretrained,weights_only=True, map_location="cpu"), strict=False)
            self.fem2.load_state_dict(torch.load(pretrained,weights_only=True, map_location="cpu"), strict=False)
        self.sem1 = SEM()
        self.sem2 = SEM()
        cs=(256, 512, 1024, 2048)
        self.m2dm = nn.ModuleList(
            [
                M2DB(in_dim=c, embed_dim=embed_dim, ffn_expand=1)
                for i, c in enumerate(cs)
            ]
        )
        # reconstruction
        self.reconstruction = nn.ModuleList()
        self.reconstruction.append(StackedCBRBlock(embed_dim * 2, embed_dim))
        self.reconstruction.append(StackedCBRBlock(embed_dim, 32))
        self.reconstruction.append(nn.Conv2d(32, 1, 1))
        
    def forward(self, data):
        rgb_extract = self.fem1(data["image"])
        depth_extract = self.fem2(data["depth"].repeat(1, 3, 1, 1))
        rgb_feats = self.sem1(rgb_extract,data["bin"])
        depth_feats = self.sem2(depth_extract,data["bin"])
        
        # to m2dm for fusion
        x = self.m2dm[3](rgb=rgb_feats[3], depth=depth_feats[3])
        x = self.m2dm[2](rgb=rgb_feats[2], depth=depth_feats[2], previous=cus_sample(x, scale_factor=2))
        x = self.m2dm[1](rgb=rgb_feats[1], depth=depth_feats[1], previous=cus_sample(x, scale_factor=2))
        x = self.m2dm[0](rgb=rgb_feats[0], depth=depth_feats[0], previous=cus_sample(x, scale_factor=2))
        # Reconstruction
        x = self.reconstruction[0](cus_sample(x, scale_factor=2))
        x = self.reconstruction[1](cus_sample(x, scale_factor=2))
        x = self.reconstruction[2](x)
        return x