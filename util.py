import re
import torch
import shutil
import logging
import torchscan
import numpy as np
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA
from tqdm import tqdm

import datasets_ws


def get_flops(model, input_shape=(480, 640)):
    """Return the FLOPs as a string, such as '22.33 GFLOPs'"""
    assert (
        len(input_shape) == 2
    ), f"input_shape should have len==2, but it's {input_shape}"
    module_info = torchscan.crawl_module(
        model, (3, input_shape[0], input_shape[1]))
    output = torchscan.utils.format_info(module_info)
    return re.findall("Floating Point Operations on forward: (.*)\n", output)[0]


def save_checkpoint(args, state, is_best, filename, suffix=""):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, f"best_model{suffix}.pth"))


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    if "model_db_state_dict" in checkpoint and checkpoint["model_db_state_dict"] is not None:
        raise ValueError("The model is trained separately. You should add separate_branch.")
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith("module"):
        state_dict = OrderedDict(
            {k.replace("module.", ""): v for (k, v) in state_dict.items()}
        )
    model.load_state_dict(state_dict)
    return model

def resume_model_separate(args, model, model_db):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        state_dict_db = checkpoint["model_db_state_dict"]
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        # state_dict = checkpoint
        raise NotImplementedError()

    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith("module"):
        state_dict = OrderedDict(
            {k.replace("module.", ""): v for (k, v) in state_dict.items()}
        )
        state_dict_db = OrderedDict(
            {k.replace("module.", ""): v for (k, v) in state_dict_db.items()}
        )
    model.load_state_dict(state_dict)
    model_db.load_state_dict(state_dict_db)
    return model, model_db

def resume_model_pix2pix(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if "model_netG_state_dict" in checkpoint:
        state_dict_G = checkpoint["model_netG_state_dict"]
    else:
        raise NotImplementedError()
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict_G.keys())[0].startswith("module"):
        state_dict_G = OrderedDict(
            {k.replace("module.", ""): v for (k, v) in state_dict_G.items()}
        )
    model.netG.load_state_dict(state_dict_G)
    return model
    
def resume_train(args, model, optimizer=None, strict=False, DA=None):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    if "model_db_state_dict" in checkpoint:
        raise ValueError("The model is trained separately. You should add separate_branch.")
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if DA is not None:
        DA.load_state_dict(checkpoint["DA_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(
        f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
        f"current_best_R@5 = {best_r5:.1f}"
    )
    # Copy best model to current save_dir
    if args.resume.endswith("last_model.pth"):
        shutil.copy(
            args.resume.replace(
                "last_model.pth", "best_model.pth"), args.save_dir
        )
    return model, optimizer, best_r5, start_epoch_num, not_improved_num

def resume_train_separate(args, model, model_db, optimizer=None, strict=False, DA=None):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model_db.load_state_dict(checkpoint["model_db_state_dict"], strict=strict)
    if DA is not None:
        DA.load_state_dict(checkpoint["DA_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(
        f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
        f"current_best_R@5 = {best_r5:.1f}"
    )
    # Copy best model to current save_dir
    if args.resume.endswith("last_model.pth"):
        shutil.copy(
            args.resume.replace(
                "last_model.pth", "best_model.pth"), args.save_dir
        )
    return model, model_db, optimizer, best_r5, start_epoch_num, not_improved_num

def compute_pca(args, model, full_features_dim):
    model = model.eval()
    pca_ds = datasets_ws.PCADataset(
        args, args.datasets_folder, args.dataset_name)
    dl = torch.utils.data.DataLoader(
        pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    logging.info("Computing PCA")
    with torch.no_grad():
        for i, images in tqdm(enumerate(dl), ncols=100):
            if i * args.infer_batch_size >= len(pca_features):
                break
            features = model(images).cpu().numpy()
            pca_features[
                i * args.infer_batch_size: (i * args.infer_batch_size) + len(features)
            ] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    return pca
