"""Ward image vectors from the SSL4EO-S12 Sentinel-2 encoder (ResNet-50, 3 color bands).

Usage: python scripts/21_ssl4eo_embed.py [--finetune N]
Without --finetune, the encoder is frozen. With --finetune N, the script continues the
contrastive pretraining on the Nepal patches for N passes, without labels.
Output: data/processed/s2_embedding_ssl4eo.parquet or s2_embedding_ssl4eo_ft.parquet.
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torchgeo.models import ResNet50_Weights, resnet50

from gorkha.paths import PROCESSED, ROOT

spec = importlib.util.spec_from_file_location("p19", Path(__file__).with_name("19_cnn_pretrain.py"))
p19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p19)

PATCHES = ROOT / "data" / "patches"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RGB = [2, 1, 0]  # bands B04, B03, B02 in the patch files
N_COMPONENTS = 16


def prepare(x: torch.Tensor, size: int) -> torch.Tensor:
    """Reflectance to the 0 to 1 range of the SSL4EO-S12 models, and a larger image."""
    x = (x[:, RGB].float() / 10000.0).clamp(0, 1)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def load_patches():
    index = pd.read_parquet(PATCHES / "s2_index.parquet")
    arrays, offset, start = [], {}, 0
    for tile in sorted(index["tile"].unique()):
        a = np.load(PATCHES / f"s2_{tile}.npy")
        offset[tile] = start
        start += len(a)
        arrays.append(a)
    data = torch.from_numpy(np.minimum(np.concatenate(arrays), 32767).astype("int16"))
    row = index["tile"].map(offset).to_numpy() + index["i"].to_numpy()
    return data, row, index


def ward_vectors(feats: np.ndarray, index: pd.DataFrame, path) -> None:
    wards = pd.read_parquet(PROCESSED / "wards.parquet", columns=["ward_id"])
    mean = pd.DataFrame(feats).groupby(index["node"].to_numpy()).mean()
    comp = PCA(N_COMPONENTS, random_state=0).fit(mean.to_numpy())
    z = comp.transform(mean.to_numpy())
    z = z / z.std(axis=0)
    out = pd.DataFrame(0.0, index=range(len(wards)), columns=[f"img_{k}" for k in range(N_COMPONENTS)])
    out.loc[mean.index] = z
    out["img_has"] = 0.0
    out.loc[mean.index, "img_has"] = 1.0
    out.insert(0, "ward_id", wards["ward_id"].to_numpy())
    out.to_parquet(path, index=False)
    print(f"{path.name}: wards with an image vector {int(out['img_has'].sum())} of {len(out)}, "
          f"variance in the {N_COMPONENTS} components "
          f"{comp.explained_variance_ratio_.sum():.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(0)
    data, row, index = load_patches()
    model = resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO)
    model.fc = nn.Identity()
    model = model.to(DEVICE)
    print(f"patches: {len(data)}, device: {DEVICE}, finetune passes: {args.finetune}", flush=True)

    if args.finetune:
        head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(inplace=True), nn.Linear(256, 64)).to(DEVICE)
        params = list(model.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        batch, t0 = 128, time.time()
        for epoch in range(args.finetune):
            model.train()
            perm = torch.randperm(len(data))
            total = 0.0
            for k in range(len(data) // batch):
                x = p19.to_float(data[perm[k * batch:(k + 1) * batch]].to(DEVICE))
                views = []
                for _ in range(2):
                    v = p19.augment(x)[:, RGB]
                    v = (torch.expm1(v * 2.0) / 10.0).clamp(0, 1)  # back to reflectance / 10000
                    views.append(F.interpolate(v, size=112, mode="bilinear", align_corners=False))
                with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                    za, zb = (F.normalize(head(model(v)), dim=1) for v in views)
                loss = p19.nt_xent(za.float(), zb.float())
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"pass {epoch + 1}: loss {total / (len(data) // batch):.3f}, {time.time() - t0:.0f} s",
                  flush=True)

    model.eval()
    feats = []
    with torch.no_grad():
        for k in range(0, len(data), 256):
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                f = model(prepare(data[k:k + 256].to(DEVICE), 160))
            feats.append(f.float().cpu().numpy())
    feats = np.concatenate(feats)[row]
    name = "s2_embedding_ssl4eo_ft.parquet" if args.finetune else "s2_embedding_ssl4eo.parquet"
    ward_vectors(feats, index, PROCESSED / name)
