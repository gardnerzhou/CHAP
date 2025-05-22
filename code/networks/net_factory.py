from networks.efficientunet import Effi_UNet
from networks.enet import ENet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_URPC, UNet_CCT,UNet_plus,DSNet,DualDecoder
import argparse
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.config import get_config
from networks.ResNet2d import ResUNet_2d


def net_factory(net_type="unet", in_chns=1, class_num=3,device= "cuda:0",args=None):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).to(device)
    elif net_type == "unetp":
        net = UNet_plus(in_chns=in_chns, class_num=class_num).to(device)
    elif net_type == "dual_student":
        net = DSNet(in_chns=in_chns, class_num=class_num,project_dim=args.projectdim,multiscale=args.ms,proxy_num=args.proxy_num).to(device)
    elif net_type == "resunet":
        net = ResUNet_2d(in_chns=in_chns, class_num=class_num).to(device)
    elif net_type == "dualdecoder":
        net = DualDecoder(in_chns=in_chns, class_num=class_num,args=args).to(device)
    else:
        net = None
    return net
