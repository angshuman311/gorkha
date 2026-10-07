"""Example of Sentinel-1 backscatter before and after the mainshock, near Chautara.

Source: Sentinel-1 GRD scenes from the Microsoft Planetary Computer (no account necessary).
The two scenes are from relative orbit 121: 2015-04-24 and 2015-05-06.
"""

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from rasterio.transform import GCPTransformer
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

from gorkha import plots
from gorkha.paths import FIGURES

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-grd/items/"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
SCENES = {
    "2015-04-24 (1 day before)": "S1A_IW_GRDH_1SSV_20150424T001054_20150424T001123_005618_007323",
    "2015-05-06 (11 days after)": "S1A_IW_GRDH_1SSV_20150506T001058_20150506T001123_005793_007722",
}
CENTER = (85.715, 27.775)  # Chautara, Sindhupalchok (longitude, latitude)
HALF = 400  # pixels of 10 m, thus the window is 8 km


def read_window(item: str) -> np.ndarray:
    href = requests.get(STAC + item, timeout=60).json()["assets"]["vv"]["href"]
    signed = requests.get(SIGN, params={"href": href}, timeout=60).json()["href"]
    with rasterio.open(signed) as src:
        gcps, _ = src.gcps
        row, col = GCPTransformer(gcps).rowcol(*CENTER)
        dn = src.read(1, window=Window(int(col) - HALF, int(row) - HALF, 2 * HALF, 2 * HALF))
    power = uniform_filter(dn.astype("float64") ** 2, size=5)  # 5 x 5 mean against speckle
    # A descending scene has east on the left. Flip it, then north is up and east is right.
    return np.fliplr(10 * np.log10(np.maximum(power, 1.0)))


if __name__ == "__main__":
    plots.style()
    images = {k: read_window(v) for k, v in SCENES.items()}
    before, after = images.values()
    # The scene geolocation is coarse. Find the pixel shift between the two windows with a
    # cross-correlation, then align the second window and remove the margins.
    a, b = before - before.mean(), after - after.mean()
    corr = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = (dy + HALF) % (2 * HALF) - HALF, (dx + HALF) % (2 * HALF) - HALF
    print("shift of the second window (rows, columns):", dy, dx)
    m = max(abs(dy), abs(dx)) + 1
    after = np.roll(after, (dy, dx), axis=(0, 1))
    images = {k: v[m:-m, m:-m] for k, v in zip(images, (before, after))}
    before, after = images.values()
    lo, hi = np.percentile(before, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), layout="constrained")
    for ax, (name, im) in zip(axes, images.items()):
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"Backscatter, {name}")
    diff = after - before
    axes[2].imshow(diff, cmap=plots.DIVERGING, vmin=-6, vmax=6)
    axes[2].set_title("Change (after minus before)")
    plots.scale_bar(fig, axes[2], plots.DIVERGING, -6, 6, "dB", shrink=0.7)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.plot([20, 120], [before.shape[0] - 25] * 2, color="#eda100", linewidth=2.5)
        ax.text(70, before.shape[0] - 40, "1 km", ha="center", fontsize=8, color="#eda100")
    fig.suptitle("Sentinel-1 radar images of the Chautara area, approximately 8 km window, VV channel, not terrain-corrected",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(FIGURES / "sentinel1_example_chautara.png")
    print("window", before.shape, "dB range", round(lo, 1), round(hi, 1),
          "| change: mean", round(float(diff.mean()), 2), "sd", round(float(diff.std()), 2))
