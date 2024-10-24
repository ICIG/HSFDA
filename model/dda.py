

import torch
from einops.layers.torch import Rearrange

from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


# Helper functions
def is_present(val):
    return val is not None

def get_default(val, default_val):
    return val if is_present(val) else default_val

def calculate_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# Helper classes
class ActivationSwish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GatedLinearUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        output, gate = x.chunk(2, dim=self.dim)
        return output * gate.sigmoid()

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=in_channels)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# Attention, feedforward, and convolution modules
class ScaledModule(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNormLayer(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., max_pos_emb=512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_query = nn.Linear(dim, inner_dim, bias=False)
        self.to_key_value = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        batch_size, device, heads, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, is_present(context)
        context = get_default(context, x)

        query, key, value = (self.to_query(x), *self.to_key_value(context).chunk(2, dim=-1))
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), (query, key, value))

        attention_scores = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale

        # Relative positional embedding
        seq = torch.arange(batch_size, device=device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        relative_pos_emb = self.rel_pos_emb(dist).to(query)
        pos_attention = einsum('b h n d, n r d -> b h n r', query, relative_pos_emb) * self.scale
        attention_scores = attention_scores + pos_attention

        if is_present(mask) or is_present(context_mask):
            mask = get_default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = get_default(context_mask, mask) if not has_context else get_default(context_mask, lambda: torch.ones(*context.shape[:2], device=device))
            mask_value = -torch.finfo(attention_scores.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            attention_scores.masked_fill_(~mask, mask_value)

        attention = attention_scores.softmax(dim=-1)
        output = einsum('b h i j, b h j d -> b h i d', attention, value)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.to_out(output)
        return self.dropout(output)

class FeedForwardLayer(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, dim * mult),
            ActivationSwish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)

# Convolution module replaced with SELayer
class ConformerBlockModule(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=31, attn_dropout=0., ff_dropout=0., conv_dropout=0., conv_causal=False):
        super().__init__()
        self.feedforward_1 = FeedForwardLayer(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attention = MultiHeadAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        self.conv = SELayerModule(channel=dim)
        self.feedforward_2 = FeedForwardLayer(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attention = PreNormLayer(dim, self.attention)
        self.feedforward_1 = ScaledModule(0.5, PreNormLayer(dim, self.feedforward_1))
        self.feedforward_2 = ScaledModule(0.5, PreNormLayer(dim, self.feedforward_2))

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.feedforward_1(x) + x
        x = self.attention(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.feedforward_2(x) + x
        x = self.final_norm(x)
        return x

# Squeeze-and-Excitation Layer (unchanged)
class SELayerModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayerModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch, channel, _ = x.size()
        y = self.avg_pool(x).view(batch, channel)
        y = self.fc(y).view(batch, channel, 1)
        x = x * y.expand_as(x)
        return x.permute(0, 2, 1)

# Main Conformer Model
class ConformerNetwork(nn.Module):
    def __init__(self, dim, *, depth, dim_head=64, heads=8, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=31, attn_dropout=0.3, ff_dropout=0., conv_dropout=0., conv_causal=False):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(ConformerBlockModule(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                conv_causal=conv_causal
            ))

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x

# Final dda model
class dda(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.transform = get_feature()

        if not args.feature == 'raw':
            self.dim = 768 if 'base' in args.size else 1024
            weight_dim = 13 if 'base' in args.size else 25
            if not args.ssl_model == 'wavlm':
                weight_dim -= 1
            if args.weighted_sum:
                self.weights = nn.Parameter(torch.ones(weight_dim))
                self.weights1 = nn.Parameter(torch.ones(weight_dim))
                self.softmax = nn.Softmax(-1)
                self.layer_norm = nn.Sequential(*[nn.LayerNorm(self.dim) for _ in range(weight_dim)])
            
        embed_size = 201 if args.feature == 'raw' else 201 + self.dim if args.feature == 'cross' else self.dim
        self.encoder = nn.Sequential(
            ConformerNetwork(dim=969,
                depth=3,
                dim_head=64,
                heads=4,
                ff_mult=4,
                conv_expansion_factor=2,
                conv_kernel_size=31,
                attn_dropout=0.,
                ff_dropout=0.,
                conv_dropout=0.
            ),
            nn.Linear(969, 256, bias=True),
            nn.Linear(256, 201, bias=True),
            nn.Sigmoid()
        )

    def forward(self, wav, layer_reps1, layer_reps2, output_wav=False, layer_norm=True):
        # Generate log1p
        (log1p, _phase), _len = self.transform(wav, ftype='log1p')

        # Generate SSL feature
        if self.args.feature != 'raw':
            if self.args.weighted_sum:
                ssl1 = torch.cat(layer_reps1, dim=2)
                ssl2 = torch.cat(layer_reps2, dim=2)
            else:
                ssl1 = layer_reps1[-1]
                ssl2 = layer_reps2[-1]

            B, T, embed_dim = ssl1.shape
            ssl1 = ssl1.repeat(1, 1, 2).reshape(B, -1, embed_dim)
            ssl1 = torch.cat((ssl1, ssl1[:, -1:].repeat(1, log1p.shape[2] - ssl1.shape[1], 1)), dim=1)

            if self.args.weighted_sum:
                normalized_layers1 = torch.split(ssl1, self.dim, dim=2)
                for i, (layer, norm_layer, weight) in enumerate(zip(normalized_layers1, self.layer_norm, self.softmax(self.weights1))):
                    if layer_norm:
                        layer = norm_layer(layer)
                    if i == 0:
                        out1 = layer * weight
                    else:
                        out1 += layer * weight

            B, T, embed_dim = ssl2.shape
            ssl2 = ssl2.repeat(1, 1, 2).reshape(B, -1, embed_dim)
            ssl2 = torch.cat((ssl2, ssl2[:, -1:].repeat(1, log1p.shape[2] - ssl2.shape[1], 1)), dim=1)

            if self.args.weighted_sum:
                normalized_layers2 = torch.split(ssl2, self.dim, dim=2)
                for i, (layer, norm_layer, weight) in enumerate(zip(normalized_layers2, self.layer_norm, self.softmax(self.weights1))):
                    if layer_norm:
                        layer = norm_layer(layer)
                    if i == 0:
                        out2 = layer * weight
                    else:
                        out2 += layer * weight

                combined_out = out1 + out2
                x = torch.cat((log1p.transpose(1, 2), combined_out), dim=2) if self.args.feature == 'cross' else combined_out
            else:
                x = torch.cat((log1p.transpose(1, 2), ssl1), dim=2) if self.args.feature == 'cross' else ssl1
        else:
            x = log1p.transpose(1, 2)

        output = self.encoder(x).transpose(1, 2)

        if self.args.target == 'IRM':
            output = output * log1p

        if output_wav:
            output = feature_to_wav((output, _phase), _len)

        return output

def create_model(args):
    return dda(args)

               

