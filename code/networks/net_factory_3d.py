from networks.unet_3D import unet_3D
from networks.vnet import VNet,DualDecoder3d
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2,mode='train',device= "cuda:0",args=None):

    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).to(device)
    elif net_type == "attention_unet":
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns).to(device)
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num).cuda()
    elif net_type == "vnet" and mode == 'train':
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).to(device)
    elif net_type == "vnet" and mode == 'test':
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=False).to(device)
    elif net_type == "dualdecoder" and mode == 'train':
        net = DualDecoder3d(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True,args=args).to(device)
    elif net_type == "dualdecoder" and mode == 'test':
        net = DualDecoder3d(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=False).to(device)
    else:
        net = None
    
    return net
