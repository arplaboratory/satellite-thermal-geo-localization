from model.functional import sare_ind, sare_joint
from model.sync_batchnorm import convert_model
from model import network
import datasets_ws
import commons
import parser
import test
import util
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import copy
import wandb
from uuid import uuid4

torch.backends.cudnn.benchmark = True  # Provides a speedup

# Initial setup: parser, logging...
args = parser.parse_arguments()
if args.use_extended_data:
    raise NotImplementedError("Please use train_extended.py")
    
start_time = datetime.now()
args.save_dir = join(
    "logs",
    args.save_dir,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{uuid4()}",
)
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
wandb.init(project="VTL", entity="xjh19971", config=vars(args))
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(
    f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs"
)

# Creation of Datasets
logging.debug(
    f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

train_ds = None
train_ds = datasets_ws.TripletsDataset(
    args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query
)

logging.info(f"Train query set: {train_ds}")

val_ds = datasets_ws.BaseDataset(
    args, args.datasets_folder, args.dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(
    args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

# Initialize model
model = network.GeoLocalizationNet(args)
model.to(args.device)
domain_classifier = None
if args.DA.startswith("DANN"):
    domain_classifier = model.create_domain_classifier(args)

if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
    if not args.resume:
        train_ds.is_inference = True
        model.aggregation.initialize_netvlad_layer(
                args, train_ds, model.backbone)
    args.features_dim *= args.netvlad_clusters

if args.separate_branch:
    logging.info('Backbone has separated branched for database and query')
    model_db = copy.deepcopy(model)
    model_db = torch.nn.DataParallel(model_db)
    if torch.cuda.device_count() >= 2:
        # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
        model_db = convert_model(model_db)
        model_db = model_db.to(args.device)

model = torch.nn.DataParallel(model)
if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    model = convert_model(model)
    model = model.to(args.device)

if domain_classifier is not None:
    domain_classifier = torch.nn.DataParallel(domain_classifier)
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    domain_classifier = convert_model(domain_classifier)
    domain_classifier = domain_classifier.to(args.device)

# Setup Optimizer and Loss
if args.aggregation == "crn":
    if domain_classifier is not None:
        raise NotImplementedError("DA for crn is not Implemented")
    crn_params = list(model.module.aggregation.crn.parameters())
    net_params = list(model.module.backbone.parameters()) + list(
        [
            m[1]
            for m in model.module.aggregation.named_parameters()
            if not m[0].startswith("crn")
        ]
    )
    if args.separate_branch:
        net_db_params = list(model_db.module.backbone.parameters()) + list(
        [
            m[1]
            for m in model_db.module.aggregation.named_parameters()
            if not m[0].startswith("crn")
        ]
    )
    if args.optim == "adam":
        if args.separate_branch:
            optimizer = torch.optim.Adam(
                [
                    {"params": crn_params, "lr": args.lr_crn_layer},
                    {"params": net_params, "lr": args.lr_crn_net},
                    {"params": net_db_params, "lr": args.lr_crn_net},
                ]
            )
        else:
            optimizer = torch.optim.Adam(
                [
                    {"params": crn_params, "lr": args.lr_crn_layer},
                    {"params": net_params, "lr": args.lr_crn_net},
                ]
            )
        logging.info("You're using CRN with Adam, it is advised to use SGD")
    elif args.optim == "sgd":
        if args.separate_branch:
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": crn_params,
                        "lr": args.lr_crn_layer,
                        "momentum": 0.9,
                        "weight_decay": 0.001,
                    },
                    {
                        "params": net_params,
                        "lr": args.lr_crn_net,
                        "momentum": 0.9,
                        "weight_decay": 0.001,
                    },
                    {
                        "params": net_db_params,
                        "lr": args.lr_crn_net,
                        "momentum": 0.9,
                        "weight_decay": 0.001,
                    },
                ]
            )
        else:
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": crn_params,
                        "lr": args.lr_crn_layer,
                        "momentum": 0.9,
                        "weight_decay": 0.001,
                    },
                    {
                        "params": net_params,
                        "lr": args.lr_crn_net,
                        "momentum": 0.9,
                        "weight_decay": 0.001,
                    },
                ]
            )
else:
    if args.optim == "adam":
        if args.separate_branch:
            if domain_classifier is not None:
                optimizer = torch.optim.Adam(list(model.parameters()) + list(model_db.parameters()) + list(domain_classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.Adam(list(model.parameters()) + list(model_db.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        else:
            if domain_classifier is not None:
                optimizer = torch.optim.Adam(list(model.parameters()) + list(domain_classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "sgd":
        if args.separate_branch:
            if domain_classifier is not None:
                optimizer = torch.optim.SGD(list(model.parameters()) + list(model_db.parameters()) + list(domain_classifier.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.001)
            else:
                optimizer = torch.optim.SGD(list(model.parameters()) + list(model_db.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.001)
        else:
            if domain_classifier is not None:
                optimizer = torch.optim.SGD(list(model.parameters()) + list(domain_classifier.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.001)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

if args.criterion == "triplet":
    criterion_triplet = nn.TripletMarginLoss(
        margin=args.margin, p=2, reduction="sum")
elif args.criterion == "sare_ind":
    criterion_triplet = sare_ind
elif args.criterion == "sare_joint":
    criterion_triplet = sare_joint

logging.info(f'Domain adapataion: {args.DA}')
if args.DA.startswith('DANN'):
    criterion_DA = torch.nn.NLLLoss(reduction='sum')

# Resume model, optimizer, and other training parameters
if args.resume:
    if args.aggregation != "crn":
        if args.separate_branch:
            (
                model,
                model_db,
                optimizer,
                best_r5,
                start_epoch_num,
                not_improved_num,
            ) = util.resume_train_separate(args, model, model_db, optimizer, DA=domain_classifier)
        else:
            (
                model,
                optimizer,
                best_r5,
                start_epoch_num,
                not_improved_num,
            ) = util.resume_train(args, model, optimizer, DA=domain_classifier)
    else:
        # CRN uses pretrained NetVLAD, then requires loading with strict=False and
        # does not load the optimizer from the checkpoint file.
        if args.separate_branch:
            model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train_separate(
            args, model, model_db, strict=False, DA=domain_classifier
        )
        else:
            model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(
            args, model, strict=False, DA=domain_classifier
        )
    logging.info(
        f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}"
    )
else:
    best_r5 = start_epoch_num = not_improved_num = 0

if args.backbone.startswith("vit"):
    logging.info(f"Output dimension of the model is {args.features_dim}")
else:
    model = model.eval()
    logging.info(
        f"Output dimension of the model is {args.features_dim}"
    )
    # logging.info(
    #     f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model, args.resize)}"
    # )

# Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")

    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    epoch_triplet_losses = np.zeros((0, 1), dtype=np.float32)
    if args.DA != 'none':
        p = epoch_num / args.epochs_num # p in [0, 1)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        epoch_DA_losses = np.zeros((0, 1), dtype=np.float32)
    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")

        # Compute triplets to use in the triplet loss
        train_ds.is_inference = True
        if args.separate_branch:
            train_ds.compute_triplets(args, model, model_db)
        else:
            train_ds.compute_triplets(args, model)
        train_ds.is_inference = False

        triplets_dl = DataLoader(
            dataset=train_ds,
            num_workers=args.num_workers,
            batch_size=args.train_batch_size,
            collate_fn=datasets_ws.collate_fn,
            pin_memory=(args.device == "cuda"),
            drop_last=True,
        )

        model = model.train()
        if args.separate_branch:
            model_db = model_db.train()

        # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=512, W=512
        # triplets_local_indexes shape: (train_batch_size*10)*3 ; because 10 triplets per query
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):

            # Flip all triplets or none
            if args.horizontal_flip:
                images = transforms.RandomHorizontalFlip()(images)

            # Compute features of all images (images contains queries, positives and negatives)
            if args.separate_branch:
                # model is for query and model_db is for database
                # query1 + pos1 + neg1s(neg_num) + query2 + pos2 + neg2(neg_num) + ...
                # Extract query image
                query_images_index = np.arange(0, len(images), 1 + 1 + args.negs_num_per_query)
                images_index = np.arange(0, len(images))
                database_images_index = np.setdiff1d(images_index, query_images_index, assume_unique=True)
                query_images = images[query_images_index]
                database_images = images[database_images_index]
                if args.DA.startswith('DANN'):
                    database_feature, database_reverse_x = model_db(database_images.to(args.device), is_train=True, alpha=alpha)
                    positive_images_index_local = np.arange(0, len(database_reverse_x), 1 + args.negs_num_per_query)
                    if args.DA_only_positive:
                        database_reverse_x = database_reverse_x[positive_images_index_local]
                    database_domain_label = domain_classifier(database_reverse_x)
                    query_feature, query_reverse_x = model(query_images.to(args.device), is_train=True, alpha=alpha)
                    query_domain_label = domain_classifier(query_reverse_x)
                else:
                    database_feature = model_db(database_images.to(args.device), is_train=True)
                    query_feature = model(query_images.to(args.device), is_train=True)
                features = torch.empty((len(images), query_feature.shape[1])).to(args.device)
                features[query_images_index] = query_feature
                features[database_images_index] = database_feature
                del database_feature, query_feature
            else:
                if args.DA.startswith('DANN'):
                    images_index = np.arange(0, len(images))
                    query_images_index = np.arange(0, len(images), 1 + 1 + args.negs_num_per_query)
                    database_images_index = np.setdiff1d(images_index, query_images_index, assume_unique=True)
                    positive_images_index = np.arange(1, len(images), 1 + 1 + args.negs_num_per_query)
                    features, reverse_x = model(images.to(args.device), is_train=True)
                    if args.DA_only_positive:
                        database_reverse_x = reverse_x[positive_images_index]
                    else:
                        database_reverse_x = reverse_x[database_images_index]
                    query_reverse_x = reverse_x[query_images_index]
                    database_domain_label = domain_classifier(database_reverse_x)
                    query_domain_label = domain_classifier(query_reverse_x)
                else:
                    features = model(images.to(args.device), is_train=True)
            loss_triplet = 0

            if args.criterion == "triplet":
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(
                        args.train_batch_size, args.negs_num_per_query, 3
                    ),
                    1,
                    0,
                )
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    loss_triplet += criterion_triplet(
                        features[queries_indexes],
                        features[positives_indexes],
                        features[negatives_indexes],
                    )
            elif args.criterion == "sare_joint":
                # sare_joint needs to receive all the negatives at once
                triplet_index_batch = triplets_local_indexes.view(
                    args.train_batch_size, 10, 3
                )
                for batch_triplet_index in triplet_index_batch:
                    q = features[batch_triplet_index[0, 0]].unsqueeze(
                        0
                    )  # obtain query as tensor of shape 1xn_features
                    p = features[batch_triplet_index[0, 1]].unsqueeze(
                        0
                    )  # obtain positive as tensor of shape 1xn_features
                    n = features[
                        batch_triplet_index[:, 2]
                    ]  # obtain negatives as tensor of shape 10xn_features
                    loss_triplet += criterion_triplet(q, p, n)
            elif args.criterion == "sare_ind":
                for triplet in triplets_local_indexes:
                    # triplet is a 1-D tensor with the 3 scalars indexes of the triplet
                    q_i, p_i, n_i = triplet
                    loss_triplet += criterion_triplet(
                        features[q_i: q_i + 1],
                        features[p_i: p_i + 1],
                        features[n_i: n_i + 1],
                    )

            del features
            loss_triplet /= args.train_batch_size * args.negs_num_per_query

            if args.DA.startswith('DANN'):
                query_target_label = torch.zeros(query_domain_label.shape[0]).long().to(args.device)
                if args.DA_only_positive:
                    # Positive sample num = query sample num
                    database_target_label = torch.ones(query_domain_label.shape[0]).long().to(args.device)
                else:
                    database_target_label = torch.ones(database_domain_label.shape[0]).long().to(args.device)
                loss_DA = criterion_DA(query_domain_label, query_target_label) + \
                          criterion_DA(database_domain_label, database_target_label)
                loss_DA /= query_domain_label.shape[0] + database_domain_label.shape[0]
                loss = loss_triplet + args.lambda_DA * loss_DA
            else:
                loss = loss_triplet

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Keep track of all losses by appending them to epoch_losses
            triplet_loss = loss_triplet.item()
            batch_loss = loss.item()
            epoch_triplet_losses = np.append(epoch_triplet_losses, triplet_loss)
            epoch_losses = np.append(epoch_losses, batch_loss)
            if args.DA != 'none':
                DA_loss = loss_DA.item()
                epoch_DA_losses = np.append(epoch_DA_losses, DA_loss)
            del loss

        debug_str = f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): "+ \
            f"current batch sum loss = {batch_loss:.4f}, "+ \
            f"average epoch sum loss = {epoch_losses.mean():.4f}, "+ \
            f"current batch triplet loss = {triplet_loss:.4f}, "+ \
            f"average epoch triplet loss = {epoch_triplet_losses.mean():.4f}, "

        if args.DA != 'none':
            debug_str+= f"current batch DA loss = {DA_loss:.4f}, "+ \
            f"average epoch DA loss = {epoch_DA_losses.mean():.4f}, "

        logging.debug(debug_str)
    
    del triplets_dl
    
    info_str = f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "+ \
        f"average epoch sum loss = {epoch_losses.mean():.4f}, "+ \
        f"average epoch triplet loss = {epoch_triplet_losses.mean():.4f}, "
    if args.DA != 'none':
        info_str += f"average epoch DA loss = {epoch_DA_losses.mean():.4f}, "

    logging.info(info_str)

    # Compute recalls on validation set
    if args.separate_branch:
        recalls, recalls_str = test.test(args, val_ds, model, model_db)
    else:
        recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

    is_best = recalls[1] > best_r5

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(
        args,
        {
            "epoch_num": epoch_num,
            "model_state_dict": model.state_dict(),
            "model_db_state_dict": model_db.state_dict() if args.separate_branch else None,
            "DA_state_dict": domain_classifier.state_dict() if domain_classifier is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "recalls": recalls,
            "best_r5": best_r5,
            "not_improved_num": not_improved_num,
        },
        is_best,
        filename="last_model.pth",
    )

    if args.DA != 'none':
        wandb.log({
                "epoch_num": epoch_num,
                "recall1": recalls[0],
                "recall5": recalls[1],
                "best_r5": recalls[1] if is_best else best_r5,
                "sum_loss": epoch_losses.mean(),
                "triplet loss": epoch_triplet_losses.mean(),
                "DA loss": epoch_DA_losses.mean()
            },)
    else:
        wandb.log({
                "epoch_num": epoch_num,
                "recall1": recalls[0],
                "recall5": recalls[1],
                "best_r5": recalls[1] if is_best else best_r5,
                "sum_loss": epoch_losses.mean(),
                "triplet loss": epoch_triplet_losses.mean(),
                "DA loss": 0
            },)

    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(
            f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}"
        )
        best_r5 = recalls[1]
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}"
        )
        if not_improved_num >= args.patience:
            logging.info(
                f"Performance did not improve for {not_improved_num} epochs. Stop training."
            )
            break


logging.info(f"Best R@5: {best_r5:.1f}")
logging.info(
    f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}"
)

# Test best model on test set
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))[
    "model_state_dict"
]
model.load_state_dict(best_model_state_dict)
if args.separate_branch:
    best_model_db_state_dict = torch.load(join(args.save_dir, "best_model.pth"))[
        "model_db_state_dict"
    ]
    model_db.load_state_dict(best_model_db_state_dict)

if args.separate_branch:
    recalls, recalls_str = test.test(
        args, test_ds, model, model_db=model_db, test_method=args.test_method)
else:
    recalls, recalls_str = test.test(
        args, test_ds, model, model_db=model, test_method=args.test_method)

wandb.log({
        "final_recall1": recalls[0],
        "final_recall5": recalls[1],
    },)
        
logging.info(f"Recalls on {test_ds}: {recalls_str}")
