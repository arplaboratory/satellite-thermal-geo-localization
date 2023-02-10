
import os
import torch
import logging
import torchvision
from torch import nn
import torchvision
from os.path import join
from transformers import ViTModel
from google_drive_downloader import GoogleDriveDownloader as gdd

from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation
from model.non_local import NonLocalBlock
from model.functional import ReverseLayerF
from model.pix2pix_networks.networks import UnetGenerator, GANLoss, NLayerDiscriminator, get_scheduler
from model.sync_batchnorm import convert_model

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
        domain_classifier = None
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
                                                    nn.Conv2d(args.features_dim * 8, 2, kernel_size=2),
                                                    nn.Flatten(),
                                                    nn.LogSoftmax(dim=1))
        elif self.DA == 'DANN_after':
            domain_classifier = nn.Sequential(nn.Linear(args.conv_output_dim, 100, bias=False),
                                                   nn.BatchNorm1d(100),
                                                   nn.ReLU(True),
                                                   nn.Linear(100, 2),
                                                   nn.LogSoftmax(dim=1))
        else:
            raise NotImplementedError()
        return domain_classifier


    def forward(self, x, is_train=False, alpha=1.0):
        x = self.backbone(x)
        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit"):
            x = x.last_hidden_state[:, 0, :]
            return x
        if hasattr(self, "conv_layer"):
            x = self.conv_layer(x)
        x_after = self.aggregation(x)
        if is_train is True:
            if self.DA == 'none':
                return x_after
            elif self.DA.startswith('DANN_before'):
                if self.DA == 'DANN_before':
                    reverse_x = ReverseLayerF.apply(x.view(x.shape[0], -1), alpha)
                elif self.DA == 'DANN_before_conv':
                    reverse_x = ReverseLayerF.apply(x, alpha)
            elif self.DA == 'DANN_after':
                reverse_x = ReverseLayerF.apply(x_after, alpha)
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
    else: raise NotImplementedError()
    
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
        self.model_name = args.G_net
        if args.G_net == 'unet':
            self.model = UnetGenerator(input_channel_num, output_channel_num, 8, norm=args.GAN_norm, upsample=args.GAN_upsample)
        else:
            raise NotImplementedError()
    
    def forward(self, x):
        x = self.model(x)
        return x
    

class pix2pix():
    def __init__(self, args, input_channel_num, output_channel_num, for_training=False):
        super().__init__()
        if args.G_net == 'unet':
            self.netG = UnetGenerator(input_channel_num, output_channel_num, 8, norm=args.GAN_norm, upsample=args.GAN_upsample, use_tanh=args.G_tanh)
        elif args.G_net == 'unet_deep':
            self.netG = UnetGenerator(input_channel_num, output_channel_num, 9, norm=args.GAN_norm, upsample=args.GAN_upsample, use_tanh=args.G_tanh)
        else:
            raise NotImplementedError()
        self.device = args.device
        if for_training:
            if args.D_net == 'patchGAN':
                self.netD = NLayerDiscriminator(input_channel_num + output_channel_num)
            elif args.D_net == 'patchGAN_deep':
                self.netD = NLayerDiscriminator(input_channel_num + output_channel_num, n_layers=4)
            else:
                raise NotImplementedError()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
            self.scheduler_G = get_scheduler(self.optimizer_G, args)
            self.scheduler_D = get_scheduler(self.optimizer_D, args)
            self.G_loss_lambda = args.G_loss_lambda
            self.criterionGAN = GANLoss(args.GAN_mode).to(args.device)
            self.criterionAUX = torch.nn.L1Loss()

    def setup(self):
        if hasattr(self, 'netD'):
            self.netD = self.init_net(self.netD)
        self.netG = self.init_net(self.netG)

    def init_net(self, model):
        model = torch.nn.DataParallel(model)
        if torch.cuda.device_count() >= 2:
            # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
            model = convert_model(model)
            model = model.to(self.device)
        return model
    
    def set_input(self, A, B):
        self.real_A = A.to(self.device)
        self.real_B = B.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionAUX(self.fake_B, self.real_B) * self.G_loss_lambda
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizer_G.param_groups[0]['lr']
        self.scheduler_G.step()
        self.scheduler_D.step()
        lr = self.optimizer_G.param_groups[0]['lr']
        logging.debug('learning rate %.7f -> %.7f' % (old_lr, lr))