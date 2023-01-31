
import os
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel
from google_drive_downloader import GoogleDriveDownloader as gdd

from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation
from model.non_local import NonLocalBlock
from model.functional import ReverseLayerF
from model.unet.unet_model import UNet

# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
    'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places'  : '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places' : '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places'     : '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2'   : '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2'   : '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2'  : '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2'      : '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
}


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)
        self.self_att = False
        self.DA = args.DA

        if args.aggregation in ["gem", "spoc", "mac", "rmac"]:
            if args.l2 == "before_pool":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":
                self.aggregation = nn.Sequential(self.aggregation, Flatten())
        
        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim
        
        if args.aggregation in ["netvlad", "crn"] and args.conv_output_dim != None:
            # Concatenate conv layer to the aggregation layer
            actual_conv_output_dim = int(args.conv_output_dim / args.netvlad_clusters)
            logging.debug(f"Last conv layer dim: {actual_conv_output_dim}")
            if args.add_bn:
                self.conv_layer = nn.Sequential(nn.Conv2d(args.features_dim, actual_conv_output_dim, 1, bias=False),
                                                nn.BatchNorm2d(actual_conv_output_dim),)
            else:
                self.conv_layer = nn.Conv2d(args.features_dim, actual_conv_output_dim, 1)
            args.features_dim = actual_conv_output_dim

        if args.non_local:
            non_local_list = [NonLocalBlock(channel_feat=get_output_channels_dim(self.backbone),
                                           channel_inner=args.channel_bottleneck)]* args.num_non_local
            self.non_local = nn.Sequential(*non_local_list)
            self.self_att = True

    def create_domain_classifier(self, args):
        if self.DA.startswith('DANN_before'):
            # Input dim = backbone_output_dim * H * W
            if self.DA == 'DANN_before':
                domain_classifier = nn.Sequential(nn.Linear(args.features_dim * 32 * 32, 1000, bias=False),
                                                    nn.BatchNorm1d(1000),
                                                    nn.ReLU(True),
                                                    nn.Linear(1000, 2),
                                                    nn.LogSoftmax(dim=1))
            elif self.DA == 'DANN_before_conv':
                domain_classifier = nn.Sequential(nn.Conv2d(args.features_dim, args.features_dim * 2, kernel_size=4, stride=2, bias=False),
                                                    nn.BatchNorm2d(args.features_dim * 2),
                                                    nn.ReLU(True),
                                                    nn.Conv2d(args.features_dim * 2, args.features_dim * 4, kernel_size=4, stride=2, bias=False),
                                                    nn.BatchNorm2d(args.features_dim * 4),
                                                    nn.ReLU(True),
                                                    nn.Conv2d(args.features_dim * 4, args.features_dim * 8, kernel_size=4, stride=2, bias=False),
                                                    nn.BatchNorm2d(args.features_dim * 8),
                                                    nn.ReLU(True),
                                                    nn.Conv2d(args.features_dim * 8, 2, kernel_size=4),
                                                    nn.LogSoftmax(dim=1))
        elif self.DA == 'DANN_after':
            domain_classifier = nn.Sequential(nn.Linear(args.conv_output_dim, 100, bias=False),
                                                   nn.BatchNorm1d(100),
                                                   nn.ReLU(True),
                                                   nn.Linear(100, 2),
                                                   nn.LogSoftmax(dim=1))
        return domain_classifier


    def forward(self, x, train=False, alpha=1.0):
        x = self.backbone(x)
        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit"):
            x = x.last_hidden_state[:, 0, :]
            return x
        if hasattr(self, "conv_layer"):
            x = self.conv_layer(x)
        x_after = self.aggregation(x)
        if train is True:
            if self.DA == 'none':
                return x_after
            elif self.DA.startswith('DANN_before'):
                if self.DA == 'DANN_before':
                    reverse_x = ReverseLayerF.apply(x.view(x.shape[0], -1), alpha)
                elif self.DA == 'DANN_before_conv':
                    reverse_x = ReverseLayerF.apply(x, alpha)
            elif self.DA == 'DANN_after':
                reverse_x = ReverseLayerF.apply(x_after.view(x_after.shape[0], -1), alpha)
            return x_after, reverse_x
        return x_after


def get_aggregation(args):
    if args.aggregation == "gem":
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "spoc":
        return aggregation.SPoC()
    elif args.aggregation == "mac":
        return aggregation.MAC()
    elif args.aggregation == "rmac":
        return aggregation.RMAC()
    elif args.aggregation == "netvlad":
        if hasattr(args, "conv_output_dim") and args.conv_output_dim is not None:
            actual_conv_output_dim = int(args.conv_output_dim / args.netvlad_clusters)
            return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=actual_conv_output_dim,
                                    work_with_tokens=args.work_with_tokens)
        else:
            return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim,
                                    work_with_tokens=args.work_with_tokens)
    elif args.aggregation == 'crn':
        if hasattr(args, "conv_output_dim") and args.conv_output_dim is not None:
            actual_conv_output_dim = int(args.conv_output_dim / args.netvlad_clusters)
            return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=actual_conv_output_dim)
        else:
            return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=args.features_dim)
    elif args.aggregation == "rrm":
        return aggregation.RRM(args.features_dim)
    elif args.aggregation == 'none'\
            or args.aggregation == 'cls' \
            or args.aggregation == 'seqpool':
        return nn.Identity()


def get_pretrained_model(args):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    
    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name +".pth")
    
    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name],
                                            dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        update_state_dict = dict()
        for key, value in state_dict.items():
            remove_prefix_key = key.replace('module.encoder.', '')
            update_state_dict[remove_prefix_key] = value
        update_state_dict.pop('fc.weight', None)
        update_state_dict.pop('fc.bias', None)
        state_dict = update_state_dict
    model.load_state_dict(state_dict, strict=False)
    return model


def get_backbone(args):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = args.backbone.startswith('cct') or args.backbone.startswith('vit')
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        if not args.unfreeze:
            for name, child in backbone.named_children():
                # Freeze layers before conv_3
                if name == "layer3":
                    break
                for params in child.parameters():
                    params.requires_grad = False
        if args.backbone.endswith("conv4"):
            if not args.unfreeze:
                logging.debug(f"Train only conv4_x of the {args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            else:
                logging.debug(f"Train only conv4_x of the {args.backbone.split('conv')[0]} (remove conv5_x)")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            if not args.unfreeze:
                logging.debug(f"Train only conv4_x and conv5_x of the {args.backbone.split('conv')[0]}, freeze the previous ones")
            else:
                logging.debug(f"Train only conv4_x and conv5_x of the {args.backbone.split('conv')[0]}")
            layers = list(backbone.children())[:-2]

        if args.remove_relu is True and (args.backbone.startswith("resnet50") or args.backbone.startswith("resnet101")):
            last_layer = layers[-1][-1]
            last_layer = nn.Sequential(*list(last_layer.modules())[1:-1])
            layers[-1][-1] = last_layer

    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        if not args.unfreeze:
            logging.debug("Train last layers of the vgg16, freeze the previous ones")
        else:
            logging.debug("Train last layers of the vgg16")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        if not args.unfreeze:
            logging.debug("Train last layers of the alexnet, freeze the previous ones")
        else:
            logging.debug("Train last layers of the alexnet")
    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone
    elif args.backbone.startswith("vit"):
        if args.resize[0] == 224:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.resize[0] == 384:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')
        else:
            raise ValueError('Image size for ViT must be either 224 or 384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return backbone

    
    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]

class GenerativeNet(nn.Module):
    def __init__(self, args, input_channel_num, output_channel_num):
        super().__init__()
        if args.generative_net == 'unet':
            self.model = UNet(input_channel_num, output_channel_num)
        else:
            raise KeyError()
    
    def forward(self, x):
        x = self.model(x)
        return x

