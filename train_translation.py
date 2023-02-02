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
torch.backends.cudnn.benchmark = True  # Provides a speedup


# Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join(
    "logs",
    args.save_dir,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
)
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
wandb.init(project="VTLG", entity="xjh19971", config=vars(args))
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(
    f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs"
)

# Creation of Datasets
logging.debug(
    f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

train_ds = None
train_ds = datasets_ws.TranslationDataset(
    args, args.datasets_folder, args.dataset_name, "train")

logging.info(f"Train query set: {train_ds}")

val_ds = datasets_ws.TranslationDataset(
    args, args.datasets_folder, args.dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.TranslationDataset(
    args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

# Initialize model
model = network.GenerativeNet(args, 3, 3)
model = model.to(args.device)

model = torch.nn.DataParallel(model)
if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    model = convert_model(model)
    model = model.to(args.device)

# Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001
    )
else:
    raise NotImplementedError()

criterion_pairs = nn.SmoothL1Loss()
    
# Resume model, optimizer, and other training parameters
if args.resume:
    (
        model,
        optimizer,
        best_psnr,
        start_epoch_num,
        not_improved_num,
    ) = util.resume_train(args, model, optimizer)
    logging.info(
        f"Resuming from epoch {start_epoch_num} with best PSNR {best_psnr:.1f}"
    )
else:
    best_psnr = start_epoch_num = not_improved_num = 0

model = model.eval()

# Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")

    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")

        # Compute pairs to use in the pair loss
        train_ds.is_inference = True
        train_ds.compute_pairs(args)
        train_ds.is_inference = False

        pairs_dl = DataLoader(
            dataset=train_ds,
            num_workers=args.num_workers,
            batch_size=args.train_batch_size,
            collate_fn=datasets_ws.collate_fn,
            pin_memory=(args.device == "cuda"),
            drop_last=True,
        )

        model = model.train()

        # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=512, W=512
        # pairs_local_indexes shape: (train_batch_size*10)*3 ; because 10 pairs per query
        for images, pairs_local_indexes, _ in tqdm(pairs_dl, ncols=100):
            # Compute features of all images (images contains queries, positives and negatives)
            query_images_index = np.arange(0, len(images), 1 + 1)
            images_index = np.arange(0, len(images))
            database_images_index = np.setdiff1d(images_index, query_images_index, assume_unique=True)
            query_images = images[query_images_index].to(args.device)
            database_images = images[database_images_index]
            output_images = model(database_images.to(args.device))
            loss_pairs = criterion_pairs(output_images, query_images)

            loss = loss_pairs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss
        debug_str = f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): "+ \
            f"current batch sum loss = {batch_loss:.4f}, "+ \
            f"average epoch sum loss = {epoch_losses.mean():.4f}, "

        logging.debug(debug_str)
    
    info_str = f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "+ \
        f"average epoch sum loss = {epoch_losses.mean():.4f}, "

    logging.info(info_str)

    # Compute rPSNR on validation set
    psnr, psnr_str = test.test_translation(args, val_ds, model)
    logging.info(f"PSNR on val set {val_ds}: {psnr_str}")

    is_best = psnr > best_psnr

    wandb.log({
            "epoch_num": epoch_num,
            "psnr": psnr,
            "best_psnr": psnr if is_best else best_psnr,
            "sum_loss": epoch_losses.mean(),
        },)

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(
        args,
        {
            "epoch_num": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "psnr": psnr,
            "best_psnr": best_psnr,
            "not_improved_num": not_improved_num,
        },
        is_best,
        filename="last_model.pth",
    )

    # If PSNR did not improve for "many" epochs, stop training
    if is_best:
        logging.info(
            f"Improved: previous best PSNR = {best_psnr:.1f}, current PSNR = {psnr:.1f}"
        )
        best_psnr = psnr
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: best PSNR = {best_psnr:.1f}, current PSNR = {psnr:.1f}"
        )
        if not_improved_num >= args.patience:
            logging.info(
                f"Performance did not improve for {not_improved_num} epochs. Stop training."
            )
            break
        

logging.info(f"Best PSNR: {best_psnr:.1f}")
logging.info(
    f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}"
)

# Test best model on test set
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))[
    "model_state_dict"
]
model.load_state_dict(best_model_state_dict)

psnr, psnr_str = test.test_translation(
    args, test_ds, model)
        
logging.info(f"PSNR on {test_ds}: {psnr_str}")
