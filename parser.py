import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmarking Visual Geolocalization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--use_extended_data",
        action="store_true",
        help="Use extended data from pix2pix",
    )
    parser.add_argument(
        "--G_test_norm",
        type=str,
        default="batch",
        choices=["batch", "instance"],
        help="Test norm for G",
    )
    parser.add_argument(
        "--G_tanh",
        action="store_true",
        help="tanh for G",
    )
    parser.add_argument(
        "--GAN_epochs_decay",
        type=int,
        default=0,
        help="lr decay epoch num",
    )
    parser.add_argument(
        "--GAN_lr_policy",
        type=str,
        default="linear",
        choices="linear",
        help="lr scheduler.",
    )
    parser.add_argument(
        "--GAN_resize",
        type=int,
        default=[512, 512],
        nargs=2,
        help="Resizing shape for images (HxW).",
    )
    parser.add_argument(
        "--GAN_mode",
        type=str,
        default="lsgan",
        choices=["vanilla", "lsgan"],
        help="Choices of GAN loss"
    )
    parser.add_argument(
        "--GAN_upsample",
        type=str,
        default="bilinear",
        choices=["convtrans", "bilinear"],
        help="Save freq for GAN"
    )
    parser.add_argument(
        "--GAN_save_freq",
        type=int,
        default=0,
        help="Save freq for GAN"
    )
    parser.add_argument(
        "--GAN_norm",
        type=str,
        default="batch",
        choices=["batch", "instance"],
        help="Norm layer in GAN"
    )
    parser.add_argument(
        "--G_contrast",
        action="store_true",
        help="G_contrast"
    )
    parser.add_argument(
        "--G_gray",
        action="store_true",
        help="G_gray"
    )
    parser.add_argument(
        "--G_loss_lambda",
        type=float,
        default=100.0,
        help="G_loss_lambda only for pix2pix"
    )
    parser.add_argument(
        "--visual_all",
        action="store_true",
        help="visual_all"
    )
    parser.add_argument(
        "--DA_only_positive",
        action="store_true",
        help="Domain adaptation only applys to positive database"
    )
    parser.add_argument(
        "--D_net",
        type=str,
        default="none",
        choices=["none", "patchGAN", "patchGAN_deep"],
        help="D_net"
    )
    parser.add_argument(
        "--G_net",
        type=str,
        default="none",
        choices=["none", "unet", "unet_deep"],
        help="G_net"
    )
    parser.add_argument(
        "--lambda_DA",
        type=float,
        default=1.0,
        help="Domain adaptation loss weight"
    )
    parser.add_argument(
        "--DA",
        type=str,
        default='none',
        choices=['none', 'DANN_before', 'DANN_after', 'DANN_before_conv'],
        help="Domain adaptation"
    )
    parser.add_argument(
        "--add_bn",
        action="store_true",
        help="Add bn to compression layers"
    )
    parser.add_argument(
        "--remove_relu",
        action="store_true",
        help="Remove last relu layer of backbone"
    )
    parser.add_argument(
        "--use_faiss_gpu",
        action="store_true",
        help="Choose if we use faiss gpu version for mining. Only work for full and partial."
    )
    parser.add_argument(
        "--prior_location_threshold",
        type=int,
        default=-1,
        help="The threshold of search region from prior knowledge for train and test. If -1, then no prior knowledge"
    )
    parser.add_argument(
        "--use_best_n",
        type=int,
        default=1,
        help="Calculate the position from weighted averaged best n. If n = 1, then it is equivalent to top 1"
    )
    parser.add_argument(
        "--separate_branch",
        action="store_true",
        help="Have two separate branches"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images",
    )
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=16,
        help="Batch size for inference (caching and testing)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="triplet",
        help="loss to be used",
        choices=["triplet", "sare_ind", "sare_joint"],
    )
    parser.add_argument(
        "--margin", type=float, default=0.1, help="margin for the triplet loss"
    )
    parser.add_argument(
        "--epochs_num", type=int, default=1000, help="number of epochs to train for"
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument(
        "--lr_crn_layer",
        type=float,
        default=5e-3,
        help="Learning rate for the CRN layer",
    )
    parser.add_argument(
        "--lr_crn_net",
        type=float,
        default=5e-4,
        help="Learning rate to finetune pretrained network when using CRN",
    )
    parser.add_argument(
        "--optim", type=str, default="adam", help="_", choices=["adam", "sgd"]
    )
    parser.add_argument(
        "--cache_refresh_rate",
        type=int,
        default=1000,
        help="How often to refresh cache, in number of queries",
    )
    parser.add_argument(
        "--queries_per_epoch",
        type=int,
        default=5000,
        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate",
    )
    parser.add_argument(
        "--negs_num_per_query",
        type=int,
        default=10,
        help="How many negatives to consider per each query in the loss",
    )
    parser.add_argument(
        "--neg_samples_num",
        type=int,
        default=1000,
        help="How many negatives to use to compute the hardest ones",
    )
    parser.add_argument(
        "--mining",
        type=str,
        default="partial",
        choices=["partial", "full", "random", "msls_weighted"],
    )
    # Model parameters
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18conv4",
        choices=[
            "alexnet",
            "vgg16",
            "resnet18conv4",
            "resnet18conv5",
            "resnet50conv4",
            "resnet50conv5",
            "resnet101conv4",
            "resnet101conv5",
            "cct384",
            "vit",
        ],
        help="_",
    )
    parser.add_argument(
        "--l2",
        type=str,
        default="before_pool",
        choices=["before_pool", "after_pool", "none"],
        help="When (and if) to apply the l2 norm with shallow aggregation layers",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="netvlad",
        choices=[
            "netvlad",
            "gem",
            "spoc",
            "mac",
            "rmac",
            "crn",
            "rrm",
            "cls",
            "seqpool",
            "none",
        ],
    )
    parser.add_argument(
        "--netvlad_clusters",
        type=int,
        default=64,
        help="Number of clusters for NetVLAD layer.",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="PCA dimension (number of principal components). If None, PCA is not used.",
    )
    parser.add_argument(
        "--num_non_local", type=int, default=1, help="Num of non local blocks"
    )
    parser.add_argument("--non_local", action="store_true", help="_")
    parser.add_argument(
        "--channel_bottleneck",
        type=int,
        default=128,
        help="Channel bottleneck for Non-Local blocks",
    )
    parser.add_argument(
        "--fc_output_dim",
        type=int,
        default=None,
        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.",
    )
    parser.add_argument(
        "--conv_output_dim",
        type=int,
        default=None,
        help="Output dimension of conv layer. If None, don't use a conv layer.",
    )
    parser.add_argument(
        "--unfreeze",
        action='store_true',
        help="Unfreeze the first few layers for backbone",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="imagenet",
        choices=["imagenet", "gldv2", "places", "none"],
        help="Select the pretrained weights for the starting network",
    )
    parser.add_argument(
        "--off_the_shelf",
        type=str,
        default="imagenet",
        choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048",
    )
    parser.add_argument(
        "--trunc_te", type=int, default=None, choices=list(range(0, 14))
    )
    parser.add_argument(
        "--freeze_te", type=int, default=None, choices=list(range(-1, 14))
    )
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to load checkpoint from, for resuming training or testing.",
    )
    # Other parameters
    parser.add_argument("--device", type=str,
                        default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--num_workers", type=int, default=8, help="num_workers for all dataloaders"
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=[512, 512],
        nargs=2,
        help="Resizing shape for images (HxW).",
    )
    parser.add_argument(
        "--test_method",
        type=str,
        default="hard_resize",
        choices=[
            "hard_resize",
            "single_query",
            "central_crop",
            "five_crops",
            "nearest_crop",
            "maj_voting",
        ],
        help="This includes pre/post-processing methods and prediction refinement",
    )
    parser.add_argument(
        "--majority_weight",
        type=float,
        default=0.01,
        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement",
    )
    parser.add_argument("--efficient_ram_testing",
                        action="store_true", help="_")
    parser.add_argument("--val_positive_dist_threshold",
                        type=int, default=50, help="_")
    parser.add_argument(
        "--train_positives_dist_threshold", type=int, default=35, help="_"
    )
    parser.add_argument(
        "--recall_values",
        type=int,
        default=[1, 5, 10, 20],
        nargs="+",
        help="Recalls to be computed, such as R@5.",
    )
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float,
                        default=None, help="_")
    parser.add_argument("--horizontal_flip", action="store_true", help="_")
    parser.add_argument("--random_resized_crop",
                        type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float,
                        default=None, help="_")
    # Paths parameters
    parser.add_argument(
        "--datasets_folder", type=str, default=None, help="Path with all datasets"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="foxtech_satellite",
        help="Relative path of the dataset",
    )
    parser.add_argument(
        "--pca_dataset_folder",
        type=str,
        default=None,
        help="Path with images to be used to compute PCA (ie: pitts30k/images/train",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="default",
        help="Folder name of the current run (saved in ./logs/)",
    )
    args = parser.parse_args()

    if args.datasets_folder == None:
        try:
            args.datasets_folder = os.environ["DATASETS_FOLDER"]
        except KeyError:
            raise Exception(
                "You should set the parameter --datasets_folder or export "
                + "the DATASETS_FOLDER environment variable as such \n"
                + "export DATASETS_FOLDER=../datasets_vg/datasets"
            )

    if args.aggregation == "crn" and args.resume == None:
        raise ValueError(
            "CRN must be resumed from a trained NetVLAD checkpoint, but you set resume=None."
        )

    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError(
            "Ensure that queries_per_epoch is divisible by cache_refresh_rate, "
            + f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}"
        )

    if torch.cuda.device_count() >= 2 and args.criterion in ["sare_joint", "sare_ind"]:
        raise NotImplementedError(
            "SARE losses are not implemented for multiple GPUs, "
            + f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss."
        )

    if args.mining == "msls_weighted" and args.dataset_name != "msls":
        raise ValueError(
            "msls_weighted mining can only be applied to msls dataset, but you're using it on {args.dataset_name}"
        )

    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if (
            args.backbone not in ["resnet50conv5", "resnet101conv5"]
            or args.aggregation != "gem"
            or args.fc_output_dim != 2048
        ):
            raise ValueError(
                "Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048"
            )

    if args.prior_location_threshold != -1 and args.prior_location_threshold <= args.val_positive_dist_threshold:
        raise ValueError(f"Prior position theshold is too small to get enough negative samples. Set it to be at least more than {args.val_positive_dist_threshold}")

    if args.use_best_n < 0:
        raise ValueError("use_best_n must be large than or equal to 0")
    
    if args.separate_branch and args.criterion in ["sare_joint", "sare_ind"]:
        raise ValueError("separate_branch currently only supports triplet loss")

    if args.separate_branch and (args.train_batch_size % torch.cuda.device_count() != 0 or args.infer_batch_size % torch.cuda.device_count() != 0):
        raise ValueError("separate_branch requires the batch size is the times of gpu number")

    if args.fc_output_dim is not None and args.conv_output_dim is not None:
        raise ValueError("fc_output_dim and conv_output_dim cannot be used at the same time")

    if args.GAN_save_freq < 0:
        raise ValueError()
    return args
