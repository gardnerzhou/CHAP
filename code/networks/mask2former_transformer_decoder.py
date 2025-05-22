# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .attention_op import MyCrossAttentionLayer
#from detectron2.config import configurable
#from detectron2.layers import Conv2d
from einops import rearrange
from .position_encoding import PositionEmbeddingSine
#from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from detectron2.config import configurable
from detectron2.utils.registry import Registry
from torch.cuda.amp import autocast
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_tf_ as trunc_normal_
from typing import List
import math


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
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

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
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

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


#@TRANSFORMER_DECODER_REGISTRY.register()
class MyTransformerDecoder(nn.Module):

    #_version = 2

    #@configurable
    def __init__(
        self,
        params,
        mask_classification=True
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        
        # config
        hidden_dim=params['hidden_dim']
        nheads=params['nheads'][::-1]
        dec_layers=params['dec_layers']
        pre_norm=params['pre_norm']
        dim_feedforward=params['dim_feedforward']
        num_queries=params['num_queries']
        num_classes=params['class_num']
        mask_dim=params['mask_dim']
        enforce_input_project=params['enforce_input_project']
        in_channels=params['in_channels']

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        # 经过seg_head前是否需要激活函数？
        self.seg_head_layers=nn.ModuleList()  # 注意力Map logits(query response) 经过seg head输出分割结果，用ground-truth深监督

        for idx in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads[idx],
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                MyCrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads[idx],
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            # self.transformer_cross_attention_layers.append(
            #                 CrossAttentionLayer(
            #                     d_model=hidden_dim,
            #                     nhead=nheads,
            #                     dropout=0.0,
            #                     normalize_before=pre_norm,
            #                 )
            #             )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.seg_head_layers.append(nn.Conv2d(num_queries, num_classes,kernel_size=1))

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding 
        self.num_feature_levels = 4
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for idx in range(self.num_feature_levels):
            # if in_channels != hidden_dim or enforce_input_project:
            #     self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
            #     weight_init.c2_xavier_fill(self.input_proj[-1])
            # else:
            #     self.input_proj.append(nn.Sequential())
            self.input_proj.append(nn.Conv2d(in_channels[idx], hidden_dim, kernel_size=1))
            weight_init.c2_xavier_fill(self.input_proj[-1])
            
        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        # if self.mask_classification:
        #       self.class_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)    # no object
    

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        size_list_att=[]

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            
            size_list.append(x[i].shape[-2:])
            size_list_att.append(x[i].shape[-2:])

            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        
        size_list_att.append(x[i].shape[-2:])
        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        attn_segmaps = []
        attn_maps=[]

        # prediction heads on learnable query features
        #outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        #predictions_class.append(outputs_class)
        #predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            
            level_index = i 
            
            #attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output,attn = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            size=size_list_att[i][0]
            
            att_logits=attn.squeeze(1)  # b c (hw)
            att_logits=rearrange(att_logits, "b c (h w) -> b c h w", h=size)
            attn_maps.append(att_logits)
            att_segmap=self.seg_head_layers[i](att_logits)
            attn_segmaps.append(att_segmap)

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            #outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            #predictions_class.append(outputs_class)
            #predictions_mask.append(outputs_mask)
            
        #assert len(predictions_class) == self.num_layers + 1

        # out = {
        #     'pred_logits': predictions_class[-1],
        #     'pred_masks': predictions_mask[-1],  
        #     'aux_outputs': self._set_aux_loss(
        #         predictions_class if self.mask_classification else None, predictions_mask
        #     )
        # }
        out = attn_segmaps

        return out,attn_maps

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        # mask features 就是 cnn decoder输出的 pixel features

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # cluster center (mask embedding)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)   # query response

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (outputs_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]




class MyTransformerDecoderV1(nn.Module):

    #_version = 2

    #@configurable
    def __init__(
        self,
        params,
        mask_classification=True
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        
        # config
        hidden_dim=params['hidden_dim']
        nheads=params['nheads']
        dec_layers=params['dec_layers']
        pre_norm=params['pre_norm']
        dim_feedforward=params['dim_feedforward']
        num_queries=params['num_queries']
        num_classes=params['class_num']
        mask_dim=params['mask_dim']
        enforce_input_project=params['enforce_input_project']
        in_channels=params['in_channels']

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        # 经过seg_head前是否需要激活函数？
        self.seg_head_layers=nn.ModuleList()  # 注意力Map logits(query response) 经过seg head输出分割结果，用ground-truth深监督

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                MyCrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=1,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            # self.transformer_cross_attention_layers.append(
            #                 CrossAttentionLayer(
            #                     d_model=hidden_dim,
            #                     nhead=nheads,
            #                     dropout=0.0,
            #                     normalize_before=pre_norm,
            #                 )
            #             )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.seg_head_layers.append(nn.Conv2d(num_queries, num_classes,kernel_size=1))

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding 
        self.num_feature_levels = 4
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for idx in range(self.num_feature_levels):
            # if in_channels != hidden_dim or enforce_input_project:
            #     self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
            #     weight_init.c2_xavier_fill(self.input_proj[-1])
            # else:
            #     self.input_proj.append(nn.Sequential())
            self.input_proj.append(nn.Conv2d(in_channels[idx], hidden_dim, kernel_size=1))
            weight_init.c2_xavier_fill(self.input_proj[-1])
            
        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        # if self.mask_classification:
        #       self.class_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)    # no object
    

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        size_list_att=[]

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            
            size_list.append(x[i].shape[-2:])
            size_list_att.append(x[i].shape[-2:])

            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        
        size_list_att.append(x[i].shape[-2:])
        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        attn_segmaps = []
        attn_maps=[]

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            
            level_index = i 
            
            #attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output,attn_logits = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            size=size_list_att[i][0]
            attn_logits=attn_logits.squeeze(1)  # b c (hw)
            attn_logits=rearrange(attn_logits, "b c (h w) -> b c h w", h=size)
            attn_maps.append(attn_logits)
            att_segmap=self.seg_head_layers[i](attn_logits)
            attn_segmaps.append(att_segmap)

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            #outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            #predictions_class.append(outputs_class)
            #predictions_mask.append(outputs_mask)
            
        #assert len(predictions_class) == self.num_layers + 1

        # out = {
        #     'pred_logits': predictions_class[-1],
        #     'pred_masks': predictions_mask[-1],  
        #     'aux_outputs': self._set_aux_loss(
        #         predictions_class if self.mask_classification else None, predictions_mask
        #     )
        # }
        out = attn_segmaps

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        # mask features 就是 cnn decoder输出的 pixel features

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # cluster center (mask embedding)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)   # query response

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (outputs_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class MyKMaXTransformerDecoder(nn.Module):

    #_version = 2

    #@configurable
    def __init__(
        self,
        params,
        mask_classification=True
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        
        # config
        hidden_dim=params['hidden_dim']
        nheads=params['nheads']
        dec_layers=params['dec_layers']
        pre_norm=params['pre_norm']
        dim_feedforward=params['dim_feedforward']
        num_queries=params['num_queries']
        num_classes=params['class_num']
        mask_dim=params['mask_dim']
        enforce_input_project=params['enforce_input_project']
        in_channels=params['in_channels']

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        # 经过seg_head前是否需要激活函数？
        self.seg_head_layers=nn.ModuleList()  # 注意力Map logits(query response) 经过seg head输出分割结果，用ground-truth深监督

        for idx in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                            CrossAttentionLayer(
                                d_model=hidden_dim,
                                nhead=nheads,
                                dropout=0.0,
                                normalize_before=pre_norm,
                            )
                        )
            
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding 
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for idx in range(self.num_feature_levels):
            # if in_channels != hidden_dim or enforce_input_project:
            #     self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
            #     weight_init.c2_xavier_fill(self.input_proj[-1])
            # else:
            #     self.input_proj.append(nn.Sequential())
            self.input_proj.append(nn.Conv2d(in_channels[idx], hidden_dim, kernel_size=1))
            weight_init.c2_xavier_fill(self.input_proj[-1])
            
        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        size_list_att=[]

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            
            size_list.append(x[i].shape[-2:])
            size_list_att.append(x[i].shape[-2:])

            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        
        size_list_att.append(x[i].shape[-2:])
        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            
            level_index = i % self.num_feature_levels
            
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            
        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],  
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        # mask features 就是 cnn decoder输出的 pixel features

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # cluster center (mask embedding)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)   # query response

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (outputs_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
        