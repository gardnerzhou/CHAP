# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
from .cross_attention import MyCrossAttention
from .position_encoding import PositionEmbeddingSine
from .mask2former_transformer_decoder import MyTransformerDecoder,MyTransformerDecoderV1
import fvcore.nn.weight_init as weight_init
import math
from thop import clever_format, profile
from torchsummary import summary
from.club import MIEstimator
from .grl import WarmStartGradientReverseLayer
from einops import rearrange, repeat
from .FilterDropout import *
from torch.distributions import Binomial

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpBlock_plus(nn.Module):
    """Upssampling followed by ConvBlock, fusion by plus"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock_plus, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2, out_channels, dropout_p)
        #self.mapping = nn.Sequential(nn.Conv2d(in_channels1 ,in_channels2, 1, bias=False), nn.BatchNorm2d(in_channels2), nn.ReLU(True))

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = x2+x1
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0,bilinear=bool(self.bilinear))
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0,bilinear=bool(self.bilinear))
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0,bilinear=bool(self.bilinear))
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0,bilinear=bool(self.bilinear))

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature, with_features=False):

        output_features = []
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output_features.append(x)

        output = self.out_conv(x)

        if with_features:
            return output,output_features[0]
        else:
            return output


class Decoder_plus(nn.Module):
    def __init__(self, params):
        super(Decoder_plus, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['up_type']
        self.project_dimension = 0
        assert (len(self.ft_chns) == 5)
        
        self.up1 = UpBlock_plus(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0,bilinear=bool(self.bilinear))
        self.up2 = UpBlock_plus(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0,bilinear=bool(self.bilinear))
        self.up3 = UpBlock_plus(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0,bilinear=bool(self.bilinear))
        self.up4 = UpBlock_plus(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0,bilinear=bool(self.bilinear))

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self, feature,with_features=False):
        
        output_features = []

        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        
        x = self.up1(x4, x3)
        output_features.append(x)
        x = self.up2(x, x2)
        output_features.append(x)
        x = self.up3(x, x1)
        output_features.append(x)
        x = self.up4(x, x0)
        output_features.append(x)

        output = self.out_conv(x)

        if with_features:
            return output,output_features
        else:
            return output


class DualDecoder(nn.Module):
    def __init__(self, in_chns, class_num, args):
        super(DualDecoder, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'
                  }
        
        params_aux = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'
                  }
        
        self.encoder = Encoder(params)
        self.decoder1 = Decoder(params)
        
        self.decoder_type = args["decoder_type"]

        if self.decoder_type == 'same':
            self.decoder2 = Decoder(params)
        elif self.decoder_type == 'plus':
            self.decoder2 = Decoder_plus(params)
        elif self.decoder_type == 'mcnet':
            self.decoder2 = Decoder(params_aux)
        
    def forward(self,x,with_feat=False,dropout=False,dropout_level=None,scores=None,comp_dropout=False):
        
        feature = self.encoder(x)
        if dropout:
            
            feature1,feature2 = perform_dropout(feature,dropout_level,scores,comp_dropout)
            output1 = self.decoder1(feature1)
            output2 = self.decoder2(feature2)
           
        else:
            output1 = self.decoder1(feature)
            output2 = self.decoder2(feature)
        if with_feat:
           return output1,output2,feature
        else:
           return output1,output2
        
class Sub_Decoder(nn.Module):
    def __init__(self, params):
        super(Sub_Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
       
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
    

    def forward(self, feature,shape=None):

        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)

        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp3_out_seg, dp2_out_seg, dp1_out_seg, dp0_out_seg


class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape=None):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp3_out_seg, dp2_out_seg, dp1_out_seg, dp0_out_seg


class Decoder_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

class UNet(nn.Module):

    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu',
                  'up_type': 1}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x,with_feats=False):

        feature = self.encoder(x)
        
        output = self.decoder(feature,with_feats)
        
        return output
    
    def foward_encoder(self,x):

        return self.encoder(x)
    
    def forward_decoder(self,x,dropout_flag=False,dropout_level=None):

        if self.training:
            x = self.perform_dropout(x,dropout_flag,dropout_level)
        
        return self.decoder(x)
        
    def perform_dropout(self,x,dropout_flag=False,level=None):
        
        feature_fp = []
        
        for idx,feat in enumerate(x):

            bs,C,H,W = feat.size()
            labeled_bs = bs // 2

            if dropout_flag and idx in level:

                feature_fp.append(torch.cat((feat,nn.Dropout2d(0.5)(feat[labeled_bs:]))))

                # channel_mask = self.scores_dropout(attn_weights.clone().detach(),0.33)   # 0.33
                # # recover
                # channel_mask = channel_mask * channel_mask.numel() / channel_mask.sum()
                # feature_fp.append(torch.cat((feat,feat[labeled_bs:] * channel_mask[labeled_bs:])))
            else:
                feature_fp.append(torch.cat((feat,feat[labeled_bs:])))
        
        return feature_fp
    
class UNet_plus(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_plus, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}
        
        self.encoder = Encoder(params)
        self.decoder = Decoder_plus(params)
        # dim_in = 16
        # feat_dim = 32
        
        self.projector = nn.AvgPool2d(4,4)
        # self.projector= nn.Sequential(
        #     #nn.AvgPool2d(4, 4),
        #     nn.Conv2d(in_channels=self.ft_chns[0], out_channels=self.project_dimension,kernel_size=1),
        #     nn.BatchNorm2d(self.project_dimension),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=self.project_dimension, out_channels=self.project_dimension,kernel_size=1)
        # )

        # self.projection_head = nn.Sequential(
        #     nn.Linear(dim_in, feat_dim),
        #     nn.BatchNorm1d(feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, feat_dim)
        # )
        # self.prediction_head = nn.Sequential(
        #     nn.Linear(feat_dim, feat_dim),
        #     nn.BatchNorm1d(feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, feat_dim)
        # )
        # for class_c in range(4):
        #     selector = nn.Sequential(
        #         nn.Linear(feat_dim, feat_dim),
        #         nn.BatchNorm1d(feat_dim),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #         nn.Linear(feat_dim, 1)
        #     )
        #     self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        # for class_c in range(4):
        #     selector = nn.Sequential(
        #         nn.Linear(feat_dim, feat_dim),
        #         nn.BatchNorm1d(feat_dim),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #         nn.Linear(feat_dim, 1)
        #     )
        #     self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

    # def forward_projection_head(self, features):
    #     return self.projection_head(features)

    # def forward_prediction_head(self, features):
    #     return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output,f = self.decoder(feature,True)
        if self.training:
            return output,f
        else:
            return output


class DSNet(nn.Module):
    def __init__(self, in_chns, class_num,project_dim,multiscale,proxy_num):
        super(DSNet, self).__init__()

        self.student1 = UNet(in_chns,class_num)
        self.student2 = UNet(in_chns,class_num)
        
        self.project_dim = project_dim

        self.ms = multiscale
        self.proxy_num = proxy_num
        
        self.att1 = MyCrossAttention(dim=self.project_dim,num_heads=2)
        self.att2 = MyCrossAttention(dim=self.project_dim,num_heads=2)

        self.shared_proxy = nn.Parameter(torch.rand(self.proxy_num, self.project_dim))
        self.independent_proxy1 = nn.Parameter(torch.rand(self.proxy_num, self.project_dim))
        self.independent_proxy2 = nn.Parameter(torch.rand(self.proxy_num, self.project_dim))

        self.CLUB = MIEstimator(self.project_dim)

        if not self.ms:

            self.projector1= nn.Sequential(
                nn.AvgPool2d(4, 4),
                nn.Conv2d(in_channels=16, out_channels=self.project_dim,kernel_size=1),
                nn.BatchNorm2d(self.project_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=self.project_dim, out_channels=self.project_dim,kernel_size=1)
            )

            self.projector2= nn.Sequential(
                nn.AvgPool2d(4, 4),
                nn.Conv2d(in_channels=16, out_channels=self.project_dim,kernel_size=1),
                nn.BatchNorm2d(self.project_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=self.project_dim, out_channels=self.project_dim,kernel_size=1)
            )

        else:
            self.projector1= nn.Sequential(
                nn.Conv2d(in_channels=240, out_channels=self.project_dim,kernel_size=1),
                nn.BatchNorm2d(self.project_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=self.project_dim, out_channels=self.project_dim,kernel_size=1)
            )

            self.projector2= nn.Sequential(
                nn.Conv2d(in_channels=240, out_channels=self.project_dim,kernel_size=1),
                nn.BatchNorm2d(self.project_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=self.project_dim, out_channels=self.project_dim,kernel_size=1)
            )

    def forward(self, x):

        outputs1,f1 = self.student1(x,True)
        outputs2,f2 = self.student2(x,True)

        if self.training:
            if self.ms:
                f1 = self.ms_fusion(f1)
                f2 = self.ms_fusion(f2)
            else:
                f1 = f1[-1]
                f2 = f2[-1]
            kv_1 = self.projector1(f1[:24]) 
            kv_1 = kv_1.flatten(-2,-1).permute(0,2,1)  # [bs,D,H,W] -> [B,L,C]
            kv_2 = self.projector2(f2[:24])
            kv_2 = kv_2.flatten(-2,-1).permute(0,2,1)
            
            out_q1,out_attn1 = self.att1(torch.cat([self.shared_proxy,self.independent_proxy1]),kv_1)
            out_q2,out_attn2 = self.att2(torch.cat([self.shared_proxy,self.independent_proxy2]),kv_2)
            dis_loss = self.calculate_DistLoss(out_q1,out_q2)
            
            return outputs1,outputs2,dis_loss
        else:
            return outputs1,outputs2
    
    @torch.no_grad()
    def forward_student1(self,x):
        
        return self.student1(x)
    
    @torch.no_grad()
    def forward_student2(self,x):

        return self.student2(x)

    def ms_fusion(self,f):
        f1,f2,f3,f4 = f

        f1 = nn.AdaptiveAvgPool2d((64,64))(f1)
        f2 = nn.AdaptiveAvgPool2d((64,64))(f2)
        f3 = nn.AdaptiveAvgPool2d((64,64))(f3)
        f4 = nn.AdaptiveAvgPool2d((64,64))(f4)

        f_cat = torch.cat([f1,f2,f3,f4],dim=1)

        return f_cat

    def calculate_DistLoss(self, q1,q2):
        # * merge
        queries_num = q1.shape[1]
        group_num = queries_num//2

        first_half = q1[:, :group_num, :]
        second_half = q1[:, group_num:, :]
        first_half_mean = first_half.mean(dim=1, keepdim=True)
        second_half_mean = second_half.mean(dim=1, keepdim=True)
        merged_q1 = torch.cat((first_half_mean, second_half_mean), dim=1)

        first_half = q2[:, :group_num, :]
        second_half = q2[:, group_num:, :]
        first_half_mean = first_half.mean(dim=1, keepdim=True)
        second_half_mean = second_half.mean(dim=1, keepdim=True)
        merged_q2 = torch.cat((first_half_mean, second_half_mean), dim=1)
        
        # calculate MI
        common_f1= merged_q1[:,0,:]
        common_f2= merged_q2[:,0,:]
        dist_f1= merged_q1[:,1,:]
        dist_f2= merged_q2[:,1,:]

        common_f = (common_f1 + common_f2) / 2
        
        # minimizing mutual information
        mimin = self.CLUB(common_f1,common_f2,dist_f1, dist_f2)
        mimin_loss = self.CLUB.learning_loss(common_f1,common_f2,dist_f1, dist_f2)

        # align common features
        # ! mse or cos_simi
        align_loss = torch.mean((common_f1-common_f2)**2)
        
        return mimin_loss + mimin


















class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)
        self.aux_decoder3 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3


class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_URPC(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(
            feature, shape)
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg


