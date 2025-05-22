import torch
import torch.nn.functional as F
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math
#from detectron2.config import configurable
#from detectron2.layers import Conv2d


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=tgt,key=memory,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        #tgt = tgt + self.dropout(tgt2)
        #tgt = self.norm(tgt)
        return tgt2

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): The number of fully-connected layers in FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels, bias=False), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims, bias=False))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

class MyCrossAttention(nn.Module):
    def __init__(self, dim, part_num=2, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5
        self.dropout = 0.1

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim, bias=False)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.ass_drop   = nn.Dropout(0.1)
        
        #indepent_p_num = part_num//2
        #self.shared_proxy = nn.Parameter(torch.rand(part_num, dim))

        self.drop_prob = 0.1
        self.layer_norms = nn.LayerNorm(dim)
        self.ffn = FFN(dim, 3*dim, dropout=self.dropout)


    def forward(self, parts,supp_feat, supp_mask=None):
        # q： query_feat: B,L,C
        # k=v: masked support feat: B,L,C
        # supp_mask: B,L,C  # 0,1
        # supp_mask = 1 - supp_mask
        k = supp_feat
        v = k.clone()
        B, _, _ = k.shape
        q_ori = parts.unsqueeze(0).repeat(B,1,1) #[B,N,C]
        _, N, C = q_ori.shape
        N_s = k.size(1)

        q = self.q_fc(q_ori).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    
        k = self.k_fc(k).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_fc(v).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #[b,n,L,c]
        
        # if supp_mask is not None:
        #     supp_mask = supp_mask.permute(0,2,1).contiguous().repeat(1, self.num_heads, 1) # [bs, nH, n]

        '''using the cosine similarity instead of dot product'''
        # shape: [B,n_h,L,c]
        attn = (q @ k.transpose(-2, -1)) * self.scale # [bs, nH, nq, ns]
        #q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        #attn = torch.einsum("bhqc,bhsc->bhqs", q, k) / 0.1 #0.1 is temprature

        # if supp_mask is not None:
        #     supp_mask = supp_mask.unsqueeze(-2).float() # [bs, nH, 1, ns]
        #     supp_mask = supp_mask * -10000.0
        #     attn = attn + supp_mask       

        #attn_out = attn.mean(1) #[B,N,HW]   # att logits
        #attn_out = F.sigmoid(attn_out)
        #attn_out = attn_out * (supp_mask.permute(0,2,1)) #B,N,HW * B 
        
        attn = attn.softmax(dim=-1) #[b,N,ns]
        attn = self.attn_drop(attn)

        attn_scores = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x =  x + q_ori
        x = self.ffn(x)
        x = self.layer_norms(x)
        # print(attn_out.shape)
        return x, attn_scores.mean(dim=1) # [b,L,C]