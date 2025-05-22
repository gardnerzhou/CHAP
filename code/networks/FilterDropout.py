import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

def mask_selection(scores, percent, wrs_flag):
    # input: scores: N
    device = scores.device

    #batch_size = scores.shape[0]
    num_neurons = scores.shape[0]
    drop_num = int(num_neurons * percent)

    if wrs_flag == 0:
        # according to scores
        threshold = torch.sort(scores, dim=1, descending=True)[0][:, drop_num]
        threshold_expand = threshold.view(batch_size, 1).expand(batch_size, num_neurons)
        mask_filters = torch.where(scores > threshold_expand, torch.tensor(1.).to(device), torch.tensor(0.).to(device))

    else:
        # add random modules
        score_max = scores.max(dim=0, keepdim=True)[0]
        score_min = scores.min(dim=0, keepdim=True)[0]
        scores = (scores - score_min) / (score_max - score_min)
        
        r = torch.rand(scores.shape).to(device)  # BxC
        key = r.pow(1. / scores)
        #key = r.pow(1. / (1- scores))
        threshold = torch.sort(key, dim=1, descending=True)[0][:, drop_num]
        #threshold_expand = threshold.view(batch_size, 1).expand(batch_size, num_neurons)
        mask_filters = torch.where(key > threshold_expand, torch.tensor(1.).to(device), torch.tensor(0.).to(device))

    mask_filters = 1 - mask_filters  # BxN
    return mask_filters

def filter_dropout_channel(scores, percent, wrs_flag):
    # scores: C
    # channel_scores = channel_scores / channel_scores.sum(dim=1, keepdim=True)
    mask_filters = mask_selection(scores, percent, wrs_flag)   # BxC

    return mask_filters


def perform_dropout(x,level=None,scores=None,comp_drop=False):

    binomial = torch.distributions.binomial.Binomial(probs=0.5)
    
    feature_fp1 = []
    feature_fp2 = []
    
    for idx,feat in enumerate(x):

        bs,dim,_,_ = feat.size()
        labeled_bs = bs // 2
        unlab_feat = feat[labeled_bs:]
        
        if idx in level:
            if scores is None:
                if comp_drop:
                    dropout_mask1 = binomial.sample((labeled_bs, dim)).to(feat.device) * 2.0
                    dropout_mask2 = 2.0 - dropout_mask1

                    dropout_mask1 = dropout_mask1.view(labeled_bs, dim, 1, 1)
                    dropout_mask2 = dropout_mask2.view(labeled_bs, dim, 1, 1)

                    perturb_feat1 = unlab_feat * dropout_mask1
                    perturb_feat2 = unlab_feat * dropout_mask2
                else:
                    perturb_feat1 = nn.Dropout2d(0.5)(unlab_feat)
                    perturb_feat2 = nn.Dropout2d(0.5)(unlab_feat)
            else:
                if torch.all(scores[idx].eq(0)):
                    perturb_feat1 = nn.Dropout2d(0.5)(unlab_feat)
                    perturb_feat2 = nn.Dropout2d(0.5)(unlab_feat)
                else:
                    activation = F.adaptive_avg_pool2d(unlab_feat,(1,1)).squeeze(-1).squeeze(-1)  # [unlab_B,C]
                    channel_mask1,channel_mask2 = scores_dropoutV2(scores[idx],activation.detach(),comp_drop,'sigmoid')
                    perturb_feat1 = channel_mask1 * unlab_feat
                    perturb_feat2 = channel_mask2 * unlab_feat

        else:
            perturb_feat1 = unlab_feat
            perturb_feat2 = unlab_feat

        feature_fp1.append(torch.cat((feat,perturb_feat1)))
        feature_fp2.append(torch.cat((feat,perturb_feat2)))

    return feature_fp1,feature_fp2

def scores_dropout(scores,percent=0.5):

    # score_max = scores.max(dim=1, keepdim=True)[0]
    # score_min = scores.min(dim=1, keepdim=True)[0]
    # scores_norm = (scores - score_min) / (score_max - score_min)

    mask_filters = filter_dropout_channel(scores=scores, percent=percent, wrs_flag=1)
    mask_filters = mask_filters.to(scores.device)  # BxCx1x1
    return mask_filters
    
def scores_dropoutV1(self,grad_sim,activation,if_comp):
    #  grad_sim: C
    # 层级自适应温度系数
    # temperature = 1.0 / (features.size(1)**0.5)  # 通道数归一化
    # drop_prob /temperature  # ! 平衡通道数量的影响，比如通道数量多，概率可能低
    scores = grad_sim.unsqueeze(0).expand(activation.size(0), activation.size(1)) * activation
    min_score = scores.min()
    max_score = scores.max()
    normalized = (scores - min_score) / (max_score - min_score + 1e-8)
    drop_probs = normalized

    drop_mask1, drop_mask2 = self.drop_based_on_prob(drop_probs,if_comp)

    return drop_mask1, drop_mask2

def scores_dropoutV2(grad_sim,activation,if_comp,type):
    #  grad_sim: C
    # 层级自适应温度系数
    # temperature = 1.0 / (features.size(1)**0.5)  # 通道数归一化
    # drop_prob /temperature  # ! 平衡通道数量的影响，比如通道数量多，概率可能低
    C = grad_sim.size(0)
    scores = grad_sim.unsqueeze(0).expand(activation.size(0), activation.size(1)) * activation
    device = activation.device
    sample_sigma = torch.std(scores,dim=1,keepdim=True)
    sample_means = torch.mean(scores,dim=1,keepdim=True)
    if type == 'gauss':
        scaling_factor = 2.0
        _z_scores = (scores - sample_means) / (sample_sigma * scaling_factor + 1e-8)
        adjusted_probs = 0.5 * (1 + torch.erf(_z_scores / math.sqrt(2)))
        adjusted_probs = torch.clamp(adjusted_probs, 0.0, 1.0).to(device)
    elif type == 'sigmoid':
        temp = 2.0
        z_scores = (scores - sample_means) / (sample_sigma + 1e-8)
        adjusted_probs = torch.sigmoid(-z_scores * temp)

    drop_mask1, drop_mask2 = drop_based_on_prob(adjusted_probs,if_comp)

    return drop_mask1, drop_mask2

def drop_based_on_prob(drop_probs,if_comp):

    if if_comp:
        branch = random.randint(0, 1)
        if branch == 0:
            mask1 = torch.bernoulli(1 - drop_probs).float()
            mask2 = torch.bernoulli(drop_probs).float()
        else:
            mask1 = torch.bernoulli(drop_probs).float()
            mask2 = torch.bernoulli(1 - drop_probs).float()
    else:
        mask1 = torch.bernoulli(1-drop_probs).float()
        mask2 = torch.bernoulli(1-drop_probs).float()

    drop_mask1 = mask1.unsqueeze(-1).unsqueeze(-1)
    drop_mask2 = mask2.unsqueeze(-1).unsqueeze(-1)

    drop_mask1 = drop_mask1 * drop_mask1.numel() / drop_mask1.sum()
    drop_mask2 = drop_mask2 * drop_mask2.numel() / drop_mask2.sum()

    return drop_mask1, drop_mask2