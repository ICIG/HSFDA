from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

import torch, pdb
import torch.nn as nn
# from unilm.wavlm.WavLM import WavLM, WavLMConfig 
from model.WavLM import WavLM, WavLMConfig 
import fairseq
from utils.util import get_feature, feature_to_wav


# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None
    ):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding

        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# class ConformerConvModule(nn.Module):
#     def __init__(
#         self,
#         dim,
#         causal = False,
#         expansion_factor = 2,
#         kernel_size = 31,
#         dropout = 0.
#     ):
#         super().__init__()

#         inner_dim = dim * expansion_factor
#         padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             Rearrange('b n c -> b c n'),
#             nn.Conv1d(dim, inner_dim * 2, 1),
#             GLU(dim=1),
#             DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
#             nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
#             Swish(),
#             nn.Conv1d(inner_dim, dim, 1),
#             Rearrange('b c n -> b n c'),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential( 
            # 16 129 969
            # Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            # Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# import torch

# x = torch.randn(4,32, 16)#batch, channel, dim

# cb = ConformerConvModule(16)
# out =cb(x)
# print("out1.shape",out.shape)
# print("111222")


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.conv1 = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.conv = SELayer(channel = dim)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x) + x    
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=1):
        super(CBAMLayer, self).__init__()

        # channel attention
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False)
        )

        # spatial attention
        # self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x =x.permute(0,2,1)
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # x = spatial_out * x
        x =x.permute(0,2,1)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x =x.permute(0,2,1)
        b, c, _= x.size()
        y = self.avg_pool(x)
        
        y = y.view(b, c)

        y = self.fc(y)
        y = y.view(b, c, 1)
        x = x * y.expand_as(x)
        x =x.permute(0,2,1)
        # return x * y.expand_as(x)
        return x
# import torch
# import torch.nn as 
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _ = x.size()
#         y = self.avg_pool(x.permute(0, 2, 1)).view(b, c)
#         y = self.fc(y).view(b, c, 1)
#         return x * y.expand_as(x)

# Conformer

class Conformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.3,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal

            ))

    def forward(self, x):

        for block in self.layers:
            x = block(x)
        return x




class FAodconv(nn.Module):
    
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.transform = get_feature()
        if not args.feature=='raw':
            self.dim = 768 if 'base' in args.size else 1024
            weight_dim = 13 if 'base' in args.size else 25
            if not args.ssl_model=='wavlm':
                weight_dim = weight_dim-1
            if args.weighted_sum:
                self.weights = nn.Parameter(torch.ones(weight_dim))
                self.softmax = nn.Softmax(-1)
                layer_norm  = []
                for _ in range(weight_dim):
                    layer_norm.append(nn.LayerNorm(self.dim))
                self.layer_norm = nn.Sequential(*layer_norm)
            
        if args.feature=='raw':
            embed = 201
        elif args.feature=='ssl':
            embed = self.dim
        else:
            embed = 201+self.dim
        # print('embed+++++++++',embed)    
        self.lstm_enc = nn.Sequential(
            # nn.Linear(embed, 256, bias=True),
            Conformer(
                dim = 768,
                depth = 3,          # 12 blocks
                dim_head = 64,
                heads = 4,
                ff_mult = 4,
                conv_expansion_factor = 2,
                conv_kernel_size = 31,
                attn_dropout = 0.,
                ff_dropout = 0.,
                conv_dropout = 0.
                ),
            nn.Linear(768, 256, bias=True),  
            # nn.Linear(1024, 512, bias=True),  
            # nn.Linear(512, 256, bias=True),  
            nn.Linear(256, 201, bias=True),
            nn.Sigmoid()
        )
        # #od卷积模块
        self.odconv = ODConv(201, 201, 15,stride=1, groups=201, padding=7,K=4,batchsize=32)
    
    def forward(self,wav,layer_reps,output_wav=False,layer_norm=True):
        
#         generate log1p
        (log1p,_phase),_len = self.transform(wav,ftype='log1p')
        # print("log1p",log1p.shape)
        # print("_phase",_phase.shape)

#         generate SSL feature
#         ssl特征在这里
        if self.args.feature!='raw':
            if self.args.weighted_sum:
                ssl = torch.cat(layer_reps,2)
            else:
                ssl = layer_reps[-1]
            B,T,embed_dim = ssl.shape
            ssl = ssl.repeat(1,1,2).reshape(B,-1,embed_dim)
            ssl = torch.cat((ssl,ssl[:,-1:].repeat(1,log1p.shape[2]-ssl.shape[1],1)),dim=1)
            if self.args.weighted_sum:
                lms  = torch.split(ssl, self.dim, dim=2)
                for i,(lm,layer,weight) in enumerate(zip(lms,self.layer_norm,self.softmax(self.weights))):
                    if layer_norm:
                        lm = layer(lm)
                    if i==0:
                        out = lm*weight
                    else:
                        out = out+lm*weight
                log1p = self.odconv(log1p)
                x    = torch.cat((log1p.transpose(1,2),out),2) if self.args.feature=='cross' else out 
            else:
                log1p = self.odconv(log1p)
                x    = torch.cat((log1p.transpose(1,2),ssl),2) if self.args.feature=='cross' else ssl 
        else:
            x = log1p.transpose(1,2)
        

        # print("x.shape",x.shape)
        # out_0 = self.lstm_enc(x)
        out = self.lstm_enc(x).transpose(1,2)
        # print("out_0",out_0.shape)
        # print("out",out.shape)
        if self.args.target=='IRM':
            out = out*log1p
        if output_wav:
            out = feature_to_wav((out,_phase),_len)
        # #OD卷积模块
        # out = self.odconv(out)
        return out
    
def MainModel(args):
    
    model = FAodconv(args)
    
    return model

import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

# 输入为 [N, C, H, W]，需要两个参数，in_planes为输特征通道数，K 为专家个数
class Sattention(nn.Module):

    def __init__(self,in_planes, C, r):#ratio = 1 // r = 1 // 16（默认）
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(C, C // r)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        # 将输入特征全局池化为 [N, C, 1, 1]
        att=self.avgpool(x)

        # 将特征转化为二维 [N, C]
        att=att.view(att.shape[0],-1) 
        # 使用 sigmoid 函数输出归一化到 [0,1] 区间
        return F.relu(self.fc(att))


# class ODConv(nn.Module):
#     def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0,
#                  groups=1,K=4,batchsize = 128):
#         super().__init__()
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.K = K
#         self.dim=80
#         self.groups = groups
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.attention = Sattention(in_planes=in_planes, C = in_planes, r = 16)
#         self.weight = nn.Parameter(torch.randn(batchsize, K, self.out_planes, self.in_planes//self.groups,
#              self.kernel_size),requires_grad=True)
#         self.mtx = nn.Parameter(torch.randn(K, 1),requires_grad=True)
#         self.fc1 = nn.Linear(in_planes // 16, kernel_size)
#         self.fc2 = nn.Linear(in_planes // 16, in_planes * 1 // self.groups)
#         self.fc3 = nn.Linear(in_planes // 16, out_planes * 1)
#         self.fc4 = nn.Linear(in_planes // 16, K * 1)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
#         self.relu = nn.ReLU()
#         #self.fc_my = nn.Linear(80, 1)
#         self.bn = nn.BatchNorm1d(out_planes)
#         self.dropout = nn.Dropout(0.3)

#         self.connected_layer = nn.Linear(in_features = 768, out_features = 768)
#     def forward(self,x):
#         # 调用 attention 函数得到归一化的权重 [N, K]
#         N,in_planels, dim = x.shape
#         # print("x的shape",x.shape)

#         att=self.attention(x)#[N, Cin // 16]
#         # print("att的shape",att.shape)

#         # print(att.shape)#torch.Size([28, 5])
#         # exit()

#         att1 = self.sigmoid(self.fc1(att))#[N, kernel_size * kernel_size]
#         # print("att1的shape",att.shape)
#         att1 = att1.reshape(att1.shape[0], self.kernel_size) #[N, kernel_size]
#         # print("att1的shape",att.shape)


#         att2 = self.sigmoid(self.fc2(att))#[N, in_planes]
#         # print("att2的shape",att2.shape)

#         # att3 = self.sigmoid(self.fc3(att))#[N, out_planes]
#         # # print("att3的shape",att3.shape)

#         att4 = self.softmax(self.fc4(att))#[N, K]
#         # print("att4的shape",att4.shape)
#         # exit()

#         Weight = torch.ones(1)

#         # print(att1.shape)
#         # print(att2.shape)
#         # print(att3.shape)
#         # print(att4.shape)
#         # exit()
#         # torch.Size([28, 15, 80])
#         # torch.Size([28, 1])
#         # torch.Size([28, 80])
#         # torch.Size([28, 4])
#         # print(self.weight.shape)#torch.Size([28, 4, 80, 1, 15, 80])->torch.Size([28, 4, 80, 1, 15])
#         # exit()
#         # print(self.weight.shape)
#         # print(att1.shape)
#         # Weight=self.weight * att1.reshape(N,1,1,1,-1)
#         # print(Weight.shape)#torch.Size([16, 4, 80, 16, 15])
#         # exit()

#         for i in range(x.shape[0]):
#             if i == 0:
#                 Weight = torch.unsqueeze(self.weight[i, :, :, :, :] * att1[i, :], 0)
#             else:

#                 Weight = torch.cat([Weight, torch.unsqueeze(self.weight[i, :, :, :, :] * att1[i, :], 0)], 0)        
#         Weight = self.dropout(Weight)
#         # print(Weight.shape)#torch.Size([28, 4, 80, 1, 15, 80])->torch.Size([28, 4, 80, 1, 15])
#         # exit()
#         Weight2 = torch.ones(1)

#         for i in range(x.shape[0]):
#             if i == 0:
#                 Weight2 = torch.unsqueeze(Weight[i, :, :, :, :] * att2[i, None, :, None], 0)
#             else:
#                 Weight2 = torch.cat([Weight2, torch.unsqueeze(Weight[i, :, :, :, :] * att2[i, None, :, None], 0)], 0)
#         Weight2 = self.dropout(Weight2)

#         # print(Weight[i, :, :, :, :, :].shape)
#         #print( att2[i, None, :, None, None].shape)
#         # print(Weight2.shape)#([28, 4, 80, 1, 15])
#         # exit()
#         # Weight3 = torch.ones(1) 
#         # # print(Weight2[i, :, :, :, :].shape)
#         # # print(att3[i, None, :, None, None,  None].shape)
#         # # exit()
#         # for i in range(x.shape[0]):
#         #     if i == 0:
#         #         Weight3 = torch.unsqueeze(Weight2[i, :, :, :, :] * att3[i, None, :, None, None], 0)
#         #     else:
#         #         Weight3 = torch.cat([Weight3, torch.unsqueeze(Weight2[i, :, :, :, :] * att3[i, None, :, None, None], 0)], 0)

#         # Weight3 = self.dropout(Weight3)
#         #print(att3[i, None, :, None, None, None].shape)
#         # print(Weight3.shape)#torch.Size([28, 4, 80, 1, 15])
#         # exit()
#         # Weight4 = torch.ones(1)
#         # for i in range(x.shape[0]):
#         #     if i == 0:
#         #         Weight4 = torch.unsqueeze(Weight3[i, :, :, :, :] * att4[i, :, None, None, None], 0)
#         #     else:
#         #         Weight4 = torch.cat([Weight4, torch.unsqueeze(Weight3[i, :, :, :, :] * att4[i, :, None, None, None], 0)], 0)
#         # Weight4 = self.dropout(Weight4)
#         # # print(att4[i, :, None, None, None, None].shape)
#         # # print(Weight4.shape)#torch.Size([28, 4, 80, 1, 15])
#         # # exit()
#         # Weight4 = torch.unsqueeze(Weight4, 5)
#         # Weight4 = Weight4.permute(0, 5, 2, 3, 4, 1)

#         # Weight4 = torch.matmul(Weight4, self.mtx)
#         #修改weight4，从这开始
#         Weight4 = torch.ones(1)
#         for i in range(x.shape[0]):
#             if i == 0:
#                 Weight4 = torch.unsqueeze(Weight2[i, :, :, :, :] * att4[i, :, None, None, None], 0)
#             else:
#                 Weight4 = torch.cat([Weight4, torch.unsqueeze(Weight2[i, :, :, :, :] * att4[i, :, None, None, None], 0)], 0)
#         Weight4 = self.dropout(Weight4)
#         # print(att4[i, :, None, None, None, None].shape)
#         # print(Weight4.shape)#torch.Size([28, 4, 80, 1, 15])
#         # exit()
#         Weight4 = torch.unsqueeze(Weight4, 5)
#         Weight4 = Weight4.permute(0, 5, 2, 3, 4, 1)

#         Weight4 = torch.matmul(Weight4, self.mtx)
#         # print("weight4",Weight4.shape)
#         #修改weight4 ，从这结束
#         # print(Weight4.shape)#torch.Size([28, 1, 80, 1, 15, 80, 1])->torch.Size([28, 1, 80, 1, 15, 1])
#         # exit()
#         x=x.contiguous().view(1, -1, dim)
#         Weight4 = Weight4.view(
#             N*self.out_planes, self.in_planes//self.groups,
#             self.kernel_size)

#         # print(x.shape)#torch.Size([1, 2240, 256])
#         # print(Weight4.shape)#torch.Size([2240, 1, 15, 80])

#         # #Weight4=self.fc_my(Weight4).squeeze(-1)
#         # print(Weight4.shape)#torch.Size([2240, 1, 15, 80])


#         output=F.conv1d(x,weight=Weight4,
#                   stride=self.stride, padding=self.padding,
#                   groups=self.groups*N)
#         # 形状恢复为 [N, C_out, H, W] 
#         #print(output.shape)

     
#         _, _, dim = output.shape
#         output=output.view(N, self.out_planes,dim)
#         output = self.relu(self.bn(output))

#         # output = self.connected_layer(output)
#         return output


class ODConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0,
                 groups=1,K=4,batchsize = 128):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.K = K
        self.dim=80
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = Sattention(in_planes=in_planes, C = in_planes, r = 16)
        self.weight = nn.Parameter(torch.randn(batchsize, K, self.out_planes, self.in_planes//self.groups,
             self.kernel_size),requires_grad=True)
        self.mtx = nn.Parameter(torch.randn(K, 1),requires_grad=True)
        self.fc1 = nn.Linear(in_planes // 16, kernel_size)
        self.fc2 = nn.Linear(in_planes // 16, in_planes * 1 // self.groups)
        self.fc3 = nn.Linear(in_planes // 16, out_planes * 1)
        self.fc4 = nn.Linear(in_planes // 16, K * 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        #self.fc_my = nn.Linear(80, 1)
        self.bn = nn.BatchNorm1d(out_planes)
        self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        # 调用 attention 函数得到归一化的权重 [N, K]
        N,in_planels, dim = x.shape
        # print(" ")
        # print("x的shape",x.shape)
        att=self.attention(x)#[N, Cin // 16]
        # print("att的shape",att.shape)
        att1 = self.sigmoid(self.fc1(att))#[N, kernel_size * kernel_size]
        # print("att1的shape",att.shape)
        att1 = att1.reshape(att1.shape[0], self.kernel_size)#[N, kernel_size]
        # print("att1的shape",att.shape)
        att2 = self.sigmoid(self.fc2(att))#[N, in_planes]
        # print("att2的shape",att2.shape)
        att3 = self.sigmoid(self.fc3(att))#[N, out_planes]
        # print("att3的shape",att3.shape)
        att4 = self.softmax(self.fc4(att))#[N, K]
        # print("att4的shape",att4.shape)
        # exit()

        # N,in_planels, dim = x.shape

        # att=self.attention(x)#[N, Cin // 16]
        # # print(att.shape)#torch.Size([28, 5])
        # # exit()

        # att1 = self.sigmoid(self.fc1(att))#[N, kernel_size * kernel_size]
        # att1 = att1.reshape(att1.shape[0], self.kernel_size)#[N, kernel_size]


        # att2 = self.sigmoid(self.fc2(att))#[N, in_planes]

        # att3 = self.sigmoid(self.fc3(att))#[N, out_planes]

        # att4 = self.softmax(self.fc4(att))#[N, K]

        Weight = torch.ones(1)
        # print("weight",Weight)

        # print(att1.shape)
        # print(att2.shape)
        # print(att3.shape)
        # print(att4.shape)
        # exit()
        # torch.Size([28, 15, 80])
        # torch.Size([28, 1])
        # torch.Size([28, 80])
        # torch.Size([28, 4])
        # print(self.weight.shape)#torch.Size([28, 4, 80, 1, 15, 80])->torch.Size([28, 4, 80, 1, 15])
        # exit()
        # print(self.weight.shape)
        # print(att1.shape)
        # Weight=self.weight * att1.reshape(N,1,1,1,-1)
        # print(Weight.shape)#torch.Size([16, 4, 80, 16, 15])
        # exit()

        for i in range(x.shape[0]):
            if i == 0:
                # print("self.weight",self.weight.shape)
                Weight = torch.unsqueeze(self.weight[i, :, :, :, :] * att1[i, :], 0)
                # print("self.weight1",Weight.shape)
            else:
                Weight = torch.cat([Weight, torch.unsqueeze(self.weight[i, :, :, :, :] * att1[i, :], 0)], 0)
                # print("self.weight2",Weight.shape)        
        Weight = self.dropout(Weight)
        # print("weight",Weight.shape)
        # print(Weight.shape)#torch.Size([28, 4, 80, 1, 15, 80])->torch.Size([28, 4, 80, 1, 15])
        # exit()
        Weight2 = torch.ones(1)

        for i in range(x.shape[0]):
            if i == 0:
                Weight2 = torch.unsqueeze(Weight[i, :, :, :, :] * att2[i, None, :, None], 0)
            else:
                Weight2 = torch.cat([Weight2, torch.unsqueeze(Weight[i, :, :, :, :] * att2[i, None, :, None], 0)], 0)
        Weight2 = self.dropout(Weight2)
        # print("weight2",Weight2.shape)

        # print(Weight[i, :, :, :, :, :].shape)
        #print( att2[i, None, :, None, None].shape)
        # print(Weight2.shape)#([28, 4, 80, 1, 15])
        # exit()
        
        #此处尝试去掉weight3，减轻权重
        Weight3 = torch.ones(1) 
        # print(Weight2[i, :, :, :, :].shape)
        # print(att3[i, None, :, None, None,  None].shape)
        # exit()
        for i in range(x.shape[0]):
            if i == 0:
                Weight3 = torch.unsqueeze(Weight2[i, :, :, :, :] * att3[i, None, :, None, None], 0)
            else:
                Weight3 = torch.cat([Weight3, torch.unsqueeze(Weight2[i, :, :, :, :] * att3[i, None, :, None, None], 0)], 0)

        Weight3 = self.dropout(Weight3)

        # print("weight3",Weight3.shape)

        # print(att3[i, None, :, None, None, None].shape)
        # print(Weight3.shape)#torch.Size([28, 4, 80, 1, 15])
        # exit()
        # 原始weight4
        Weight4 = torch.ones(1)
        for i in range(x.shape[0]):
            if i == 0:
                Weight4 = torch.unsqueeze(Weight3[i, :, :, :, :] * att4[i, :, None, None, None], 0)
            else:
                Weight4 = torch.cat([Weight4, torch.unsqueeze(Weight3[i, :, :, :, :] * att4[i, :, None, None, None], 0)], 0)
        Weight4 = self.dropout(Weight4)
        # print(att4[i, :, None, None, None, None].shape)
        # print(Weight4.shape)#torch.Size([28, 4, 80, 1, 15])
        # exit()
        Weight4 = torch.unsqueeze(Weight4, 5)
        Weight4 = Weight4.permute(0, 5, 2, 3, 4, 1)

        Weight4 = torch.matmul(Weight4, self.mtx)
        # print("weight4",Weight4.shape)

        # #修改weight4 ，从这开始
        # Weight4 = torch.ones(1)
        # for i in range(x.shape[0]):
        #     if i == 0:
        #         Weight4 = torch.unsqueeze(Weight2[i, :, :, :, :] * att4[i, :, None, None, None], 0)
        #     else:
        #         Weight4 = torch.cat([Weight4, torch.unsqueeze(Weight2[i, :, :, :, :] * att4[i, :, None, None, None], 0)], 0)
        # Weight4 = self.dropout(Weight4)
        # # print(att4[i, :, None, None, None, None].shape)
        # # print(Weight4.shape)#torch.Size([28, 4, 80, 1, 15])
        # # exit()
        # Weight4 = torch.unsqueeze(Weight4, 5)
        # Weight4 = Weight4.permute(0, 5, 2, 3, 4, 1)

        # Weight4 = torch.matmul(Weight4, self.mtx)
        # print("weight4",Weight4.shape)
        # #修改weight4 ，从这结束

        # print(Weight4.shape)#torch.Size([28, 1, 80, 1, 15, 80, 1])->torch.Size([28, 1, 80, 1, 15, 1])
        # exit()
        x=x.view(1, -1, dim)
        Weight4 = Weight4.view(
            N*self.out_planes, self.in_planes//self.groups,
            self.kernel_size)
        # print("weight4的新形状",Weight4.shape)
        

        # print(x.shape)#torch.Size([1, 2240, 256])
        # print(Weight4.shape)#torch.Size([2240, 1, 15, 80])

        # #Weight4=self.fc_my(Weight4).squeeze(-1)
        # print(Weight4.shape)#torch.Size([2240, 1, 15, 80])

        output=F.conv1d(x,weight=Weight4,
                  stride=self.stride, padding=self.padding,
                  groups=self.groups*N)
        # 形状恢复为 [N, C_out, H, W] 
        #print(output.shape)
 
        _, _, dim = output.shape
        output=output.view(N, self.out_planes,dim)
        output = self.relu(self.bn(output))
        return output
if __name__ == "__main__":
    import torch

    # x = torch.randn(16,80, 272)#batch, channel, dim

    x = torch.randn(16,129, 768)#batch, channel, dim

    # conv = nn.Conv1d(201, 201, 129,stride=1,padding=7, groups=1)
    odconv = ODConv(129, 129, 15,stride=1, groups=129, padding=7,K=4,batchsize=16)

    out1 = odconv(x)
    print("out1",out1.shape)
    # print("111222")

    # print(odconv)
    # print("parameters:", sum(p.numel() for p in odconv.parameters() if p.requires_grad))#10760360

    # print(conv)
    # print("parameters:", sum(p.numel() for p in conv.parameters() if p.requires_grad))#1280

    # out3=conv(x)
    # print("out",out3.shape)


