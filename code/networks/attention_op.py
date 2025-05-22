import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MyCrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead,dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.multihead_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout)

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
        tgt2,att_logits = self.multihead_attn(q=self.with_pos_embed(tgt, query_pos),
                                   k=self.with_pos_embed(memory, pos),
                                   v=memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt,att_logits

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
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



class   MyMultiheadAttention(nn.Module):
    ''' My Multi-Head Attention module '''

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model
        self.d_v = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_model, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_model, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_model, bias=False)
        self.fc = nn.Linear(n_head * d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_model ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # batch size first
        q=q.transpose(0,1)
        k=k.transpose(0,1)
        v=v.transpose(0,1)
        
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        if self.n_head>1:
            attn = torch.mean(attn,dim=1,keepdim=True)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        # change to len_q x b x D
        q=q.transpose(0,1)

        #q += residual

        #q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        
        #corr=F.cosine_similarity(q,k,dim=3)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn_logits=attn
        attn = self.dropout(F.softmax(attn, dim=-1))
        #attn = torch.argmax(attn,dim=2)   # cluster-wise argmax
        #out = F.one_hot (attn).permute(0,1,3,2)*1.
        #out=torch.zeros_like(attn_logits).scatter_(2, attn.unsqueeze(1), 1.)
        output = torch.matmul(attn, v)

        return output, attn_logits