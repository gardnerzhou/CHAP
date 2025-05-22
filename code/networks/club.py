
import torch
import torch.nn as nn

class CLUBMean(nn.Module):  # Set variance of q(y|x) to 1, logvar = 0. Update 11/26/2022
    def __init__(self, x_dim, y_dim, hidden_size=512):
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))

        super(CLUBMean, self).__init__()

        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            self.p_mu = nn.Sequential(nn.Linear(x_dim, int(hidden_size)),
                                      nn.ReLU(),
                                      nn.Linear(int(hidden_size), y_dim))

    def get_mu_logvar(self, x_samples):
        # variance is set to 1, which means logvar=0
        mu = self.p_mu(x_samples)
        return mu, 0

    def forward(self, x_samples, y_samples):

        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2.

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


'''MI Estimator network is seperated from the main network'''
class MIEstimator(nn.Module):
    """
    f1_s为第一个branch的specific特征,f1_c为common特征
    """
    def __init__(self, dim = 128):
        super(MIEstimator, self).__init__()
        self.dim = dim
        self.mimin_glob = CLUBMean(self.dim*2, self.dim) # Can also use CLUBEstimator, but CLUBMean is more stable
        self.mimin = CLUBMean(self.dim, self.dim)
    
    def forward(self, f1_c,f2_c,f1_s, f2_s):
        mimin = self.mimin(f1_s, f2_s)
        #mimin += self.mimin_glob(torch.cat((f1_s, f2_s), dim=1), global_embed)
        #mimin -= self.mimin(f1_c,f2_c)  # 增大互信息，align
        return mimin
    
    def learning_loss(self, f1_c,f2_c,f1_s, f2_s):
        mimin_loss = self.mimin.learning_loss(f1_s, f2_s)
        #mimin_loss += self.mimin_glob.learning_loss(torch.cat((f1_s, f2_s), dim=1), global_embed).mean()
        #mimin_loss -= self.mimin.learning_loss(f1_c,f2_c) # 增大互信息，align

        return mimin_loss