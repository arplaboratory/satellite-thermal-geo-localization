"""SGM Demo: Feature extraction and thermal-to-satellite retrieval.
Loads pretrained SGM, extracts features from thermal query and satellite
database patches, and performs retrieval with visualization.
Preprocessing matches STHN/global_pipeline/datasets_ws.py.
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

# Preprocessing matching STHN/global_pipeline/datasets_ws.py
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

query_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    base_transform,
])


def make_sgm_args():
    return argparse.Namespace(
        backbone="resnet50conv4",
        aggregation="gem",
        netvlad_clusters=64,
        fc_output_dim=4096,
        l2="before_pool",
        non_local=False,
        num_non_local=1,
        channel_bottleneck=128,
        DA=False,
        pretrain="imagenet",
        off_the_shelf="imagenet",
        work_with_tokens=False,
        trunc_te=None,
        freeze_te=None,
        unfreeze=False,
        resize=[512, 512],
        device="cuda" if torch.cuda.is_available() else "cpu",
        G_contrast="manual",
    )


def download_sgm_model():
    """Download SGM model from HuggingFace if not available locally."""
    local_path = "satellite_0_thermalmapping_135_contrast_dense_exclusion-2024-02-14_23-02-31-91400d55-5881-48e5-b6cb-cecff4f47a3f/best_model.pth"
    if os.path.exists(local_path):
        logging.info(f"Using local checkpoint: {local_path}")
        return local_path
    logging.info("Downloading SGM model from HuggingFace...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="xjh19972/SGM", filename="best_model.pth")
    logging.info(f"Downloaded to: {path}")
    return path


def load_sgm_model(model_path):
    from model import network
    args = make_sgm_args()
    model = network.GeoLocalizationNet(args)
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if list(state_dict.keys())[0].startswith("module"):
        state_dict = OrderedDict(
            {k.replace("module.", ""): v for (k, v) in state_dict.items()}
        )
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    return model, args


if __name__ == "__main__":
    output_dir = "examples/sgm"

    sgm_model_path = download_sgm_model()

    # 1. Load images
    logging.info("Loading images...")
    thermal_query = Image.open(LOCAL_THERMAL).convert("RGB")
    sat_map = np.array(Image.open(LOCAL_SAT).convert("RGB"))
    logging.info(f"Thermal query: {thermal_query.size}, Satellite map: {sat_map.shape}")

    # 2. Crop database patches across the full 1536x1536 map with stride=35
    h, w = sat_map.shape[:2]
    half = 256
    satellite_patches = []
    for y in range(half, h - half, 35):
        for x in range(half, w - half, 35):
            patch = sat_map[y - half:y + half, x - half:x + half]
            satellite_patches.append((Image.fromarray(patch), y, x))
    # Ensure GT patch at center (768,768) is included
    gt_cy, gt_cx = h // 2, w // 2
    gt_exists = any(py == gt_cy and px == gt_cx for _, py, px in satellite_patches)
    if not gt_exists:
        gt_patch = sat_map[gt_cy - half:gt_cy + half, gt_cx - half:gt_cx + half]
        satellite_patches.append((Image.fromarray(gt_patch), gt_cy, gt_cx))
    logging.info(f"Cropped {len(satellite_patches)} satellite patches (includes GT at ({gt_cy},{gt_cx}))")
    del sat_map

    # 3. Load SGM
    sgm_model, sgm_args = load_sgm_model(sgm_model_path)
    logging.info(f"SGM params: {sum(p.numel() for p in sgm_model.parameters()):,}")
    logging.info(f"SGM features_dim: {sgm_args.features_dim}")

    # 4. Extract features (matching STHN/global_pipeline preprocessing)
    os.makedirs(output_dir, exist_ok=True)

    logging.info("=== SGM: Feature Extraction & Retrieval ===")
    with torch.no_grad():
        # Query: contrast -> Grayscale -> ToTensor -> Normalize -> Resize
        thermal_contrasted = transforms.functional.adjust_contrast(thermal_query, contrast_factor=3)
        query_tensor = query_transform(thermal_contrasted)
        query_tensor = transforms.functional.resize(query_tensor, sgm_args.resize)
        query_feat = sgm_model(query_tensor.unsqueeze(0).to(sgm_args.device)).cpu()

        # Database: ToTensor -> Normalize -> Resize
        db_features = []
        for patch, py, px in satellite_patches:
            db_tensor = base_transform(patch)
            db_tensor = transforms.functional.resize(db_tensor, sgm_args.resize)
            feat = sgm_model(db_tensor.unsqueeze(0).to(sgm_args.device))
            db_features.append(feat.cpu())
        db_features = torch.cat(db_features, dim=0)

    logging.info(f"Database features: {db_features.shape}")
    logging.info(f"Query feature: {query_feat.shape}, norm={torch.norm(query_feat, dim=1).item():.4f}")

    # 5. Retrieval
    distances = torch.cdist(query_feat, db_features, p=2).squeeze(0)
    sorted_indices = torch.argsort(distances)

    logging.info("\nRetrieval Results (top 5):")
    for rank, idx in enumerate(sorted_indices[:5]):
        _, py, px = satellite_patches[idx]
        logging.info(f"  Rank {rank+1}: patch {idx.item()} center=({py},{px}), L2={distances[idx].item():.4f}")

    # 6. Save visualization: query | top-1 | top-2 | top-3
    thermal_query.save(os.path.join(output_dir, "query_thermal.png"))
    thermal_contrasted.save(os.path.join(output_dir, "query_thermal_contrasted.png"))

    n_show = min(3, len(sorted_indices))
    combined = Image.new("RGB", (512 * (1 + n_show) + 10 * n_show, 512 + 40), (255, 255, 255))
    combined.paste(thermal_query.convert("RGB").resize((512, 512)), (0, 40))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((160, 8), "Thermal Query", fill="black", font=font)

    for i in range(n_show):
        idx = sorted_indices[i]
        patch, py, px = satellite_patches[idx]
        x_offset = 512 * (i + 1) + 10 * (i + 1)
        combined.paste(patch.resize((512, 512)), (x_offset, 40))
        d = distances[idx].item()
        draw.text((x_offset + 100, 8), f"Rank {i+1} ({py},{px}) L2={d:.3f}", fill="black", font=font)

    combined.save(os.path.join(output_dir, "sgm_retrieval.png"))

    logging.info(f"\nRetrieval visualization saved to {output_dir}/sgm_retrieval.png")
    logging.info("SGM demo complete!")
