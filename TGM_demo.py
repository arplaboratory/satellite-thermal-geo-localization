"""TGM Demo: Generate synthetic thermal images from satellite patches.
Downloads real satellite map from HuggingFace, crops 512x512 patches
from the valid region, and runs pix2pix TGM inference.
"""
import torch
import argparse
import logging
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict

Image.MAX_IMAGE_PIXELS = None
logging.basicConfig(level=logging.INFO)

LOCAL_SAT = "examples/satellite_1536.png"
LOCAL_THERMAL = "examples/thermal_512.png"


def make_tgm_args():
    return argparse.Namespace(
        G_net="unet",
        D_net="patchGAN",
        GAN_norm="batch",
        GAN_upsample="bilinear",
        GAN_mode="lsgan",
        G_tanh=False,
        G_loss_lambda=100.0,
        GAN_resize=[512, 512],
        lr=0.0002,
        epochs_num=200,
        GAN_epochs_decay=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def crop_patches(map_np, crop_size=512, stride=35, max_patches=20, region=None):
    """Crop crop_size x crop_size patches from a map with given stride.
    If region is given as [top, left, bottom, right], only crop within that area."""
    h, w = map_np.shape[:2]
    half = crop_size // 2
    if region is not None:
        y_start = max(half, region[0])
        x_start = max(half, region[1])
        y_end = min(h - half, region[2])
        x_end = min(w - half, region[3])
    else:
        y_start, x_start = half, half
        y_end, x_end = h - half, w - half
    patches = []
    for y in range(y_start, y_end, stride):
        for x in range(x_start, x_end, stride):
            patch = map_np[y - half:y + half, x - half:x + half]
            if patch.shape[0] == crop_size and patch.shape[1] == crop_size:
                patches.append((Image.fromarray(patch), y, x))
                if len(patches) >= max_patches:
                    return patches
    return patches


def load_tgm_model(model_path):
    """Load TGM following STHN/global_pipeline pipeline:
    1. Create pix2pix model
    2. Load weights via resume_model_pix2pix (strips module. prefix)
    3. Call setup() to wrap in DataParallel
    4. Set netG to eval mode
    """
    from model import network
    from model.sync_batchnorm import convert_model
    args = make_tgm_args()
    model = network.pix2pix(args, 3, 1)
    # Load weights (same as util.resume_model_pix2pix)
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    state_dict_G = checkpoint["model_netG_state_dict"]
    if list(state_dict_G.keys())[0].startswith("module"):
        state_dict_G = OrderedDict(
            {k.replace("module.", ""): v for (k, v) in state_dict_G.items()}
        )
    model.netG.load_state_dict(state_dict_G)
    # setup() wraps in DataParallel (same as STHN)
    model.setup()
    # eval mode (same as test_translation_pix2pix)
    model.netG = model.netG.eval()
    return model, args


if __name__ == "__main__":
    output_dir = "examples/tgm"

    # Download model from HuggingFace if not available locally
    tgm_model_path = "TGM_nocontrast/best_model.pth"
    if not os.path.exists(tgm_model_path):
        logging.info("Downloading TGM model from HuggingFace...")
        from huggingface_hub import hf_hub_download
        tgm_model_path = hf_hub_download(repo_id="xjh19972/TGM", filename="best_model.pth")
        logging.info(f"Downloaded to: {tgm_model_path}")
    else:
        logging.info(f"Using local checkpoint: {tgm_model_path}")

    # 1. Load local example images (1536x1536 satellite, 512x512 thermal)
    logging.info("Loading example images...")
    sat_map = np.array(Image.open(LOCAL_SAT).convert("RGB"))
    gt_thermal = Image.open(LOCAL_THERMAL).convert("RGB")
    logging.info(f"Satellite map: {sat_map.shape}, Thermal: {gt_thermal.size}")

    # 2. Crop one 512x512 satellite patch at center (768, 768) matching the thermal
    cy, cx = 768, 768
    half = 256
    patch = sat_map[cy - half:cy + half, cx - half:cx + half]
    satellite_patches = [(Image.fromarray(patch), cy, cx)]
    logging.info(f"Cropped {len(satellite_patches)} satellite patches")
    del sat_map

    # 5. Load TGM and run inference
    tgm_model, tgm_args = load_tgm_model(tgm_model_path)
    logging.info(f"TGM Generator params: {sum(p.numel() for p in tgm_model.netG.parameters()):,}")

    os.makedirs(output_dir, exist_ok=True)
    gan_resize = tgm_args.GAN_resize

    # Satellite input: Resize -> ToTensor -> Normalize(0.5, 0.5)
    # Same as TranslationDataset.resized_transform
    sat_transform = transforms.Compose([
        transforms.Resize(gan_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])
    # Thermal GT: Grayscale(1) -> Resize -> ToTensor -> Normalize(0.5, 0.5)
    # Same as TranslationDataset.query_transform
    thermal_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(gan_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    logging.info("=== TGM: Satellite -> Thermal Generation ===")
    with torch.no_grad():
        for i, (patch, cy, cx) in enumerate(satellite_patches):
            # Follow STHN pipeline: set_input(database=satellite, query=thermal) -> forward()
            database_tensor = sat_transform(patch).unsqueeze(0)
            query_tensor = thermal_transform(gt_thermal).unsqueeze(0)
            tgm_model.set_input(database_tensor, query_tensor)
            tgm_model.forward()
            output = tgm_model.fake_B
            output = torch.clamp(output, min=-1, max=1)
            output_img = output * 0.5 + 0.5

            out_pil = transforms.ToPILImage()(output_img[0].cpu())
            out_pil.save(os.path.join(output_dir, f"tgm_patch_{i}_y{cy}_x{cx}.png"))
            patch.save(os.path.join(output_dir, f"sat_patch_{i}_y{cy}_x{cx}.png"))
            logging.info(f"  Patch {i}: center=({cy},{cx})")

    # 6. Save ground truth and comparison
    gt_thermal.save(os.path.join(output_dir, "query_thermal.png"))

    # Side-by-side comparison for patch 0
    sat_patch = satellite_patches[0][0].resize((512, 512))
    cy, cx = satellite_patches[0][1], satellite_patches[0][2]
    tgm_patch = Image.open(os.path.join(output_dir, f"tgm_patch_0_y{cy}_x{cx}.png")).resize((512, 512))
    gt_resized = gt_thermal.convert("RGB").resize((512, 512))

    combined = Image.new("RGB", (512 * 3 + 20, 512 + 40), (255, 255, 255))
    combined.paste(sat_patch, (0, 40))
    combined.paste(tgm_patch, (522, 40))
    combined.paste(gt_resized, (1044, 40))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    draw.text((176, 8), "Satellite Input", fill="black", font=font)
    draw.text((688, 8), "TGM Generated", fill="black", font=font)
    draw.text((1214, 8), "Ground Truth", fill="black", font=font)
    combined.save(os.path.join(output_dir, "tgm_comparison.png"))

    logging.info(f"\nTGM generated {len(satellite_patches)} thermal images in {output_dir}/")
    logging.info(f"Comparison saved to {output_dir}/tgm_comparison.png")
