import copy 
import torch
import torch.nn.functional as F
from torch import nn
from einops.einops import rearrange
from torch.nn import MultiheadAttention

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    
class Crossatt_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model = 96,
                 nhead = 8,
                 no_flash=False,
                 rope=False,
                 npe=None,
                 fp32=False,
                 ):
        super(Crossatt_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        #self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)      
        #print(self.dim, self.nhead)  
        self.attention = MultiheadAttention(d_model, self.nhead, batch_first=True)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggragate feature
        assert x_mask is None and source_mask is None
        query, source = self.norm1(x.permute(0,2,3,1)), self.norm1(source.permute(0,2,3,1)) # [N, H, W, C]

        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)


        # multi-head attention handle padding mask
        query = rearrange(query, 'b h w c -> (b h) w c', h = H0, w = W0)
        key = rearrange(key, 'b h w c -> (b h) w c', h = H1, w = W1)
        value = rearrange(value, 'b h w c -> (b h) w c', h = H1, w = W1)
    
        
        m, _ = self.attention(query, key, value)  # N(b h), L(w), C
        m = self.merge(m) # [N, L, C]

        # Upsample feature
        m = rearrange(m, '(b h) w c -> b c h w', b = bs, h=H0, w=W0) # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m
    
class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self):
        super(LocalFeatureTransformer, self).__init__()
        
        self.fp32 = True 

        self.d_model = 96
        self.nhead = 8
        self.layer_names = ['cross'] * 2

        self.resid = ResidualBlock(96, 96, norm_fn = 'group')
        
        cross_layer = Crossatt_EncoderLayer(self.d_model, self.nhead,
                                            False, False, [832, 832, 832, 832], self.fp32)
        self.layers = nn.ModuleList([copy.deepcopy(cross_layer) for _ in self.layer_names])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        H0, W0, H1, W1 = feat0.size(-2), feat0.size(-1), feat1.size(-2), feat1.size(-1)
        bs = feat0.shape[0]

        feature_cropped = False
        if bs == 1 and mask0 is not None and mask1 is not None:
            mask_H0, mask_W0, mask_H1, mask_W1 = mask0.size(-2), mask0.size(-1), mask1.size(-2), mask1.size(-1)
            mask_h0, mask_w0, mask_h1, mask_w1 = mask0[0].sum(-2)[0], mask0[0].sum(-1)[0], mask1[0].sum(-2)[0], mask1[0].sum(-1)[0]
            mask_h0, mask_w0, mask_h1, mask_w1 = mask_h0//self.agg_size0*self.agg_size0, mask_w0//self.agg_size0*self.agg_size0, mask_h1//self.agg_size1*self.agg_size1, mask_w1//self.agg_size1*self.agg_size1
            feat0 = feat0[:, :, :mask_h0, :mask_w0]
            feat1 = feat1[:, :, :mask_h1, :mask_w1]
            feature_cropped = True

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):

            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)                
            else:
                raise KeyError

        feat0 = self.resid(feat0)
        feat1 = self.resid(feat1)
        return feat0, feat1
