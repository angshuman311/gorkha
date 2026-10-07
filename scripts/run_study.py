"""Driver of the full study. It runs the steps one after the other and writes a log.

Steps:
1. Wait for the ASF download and the road travel times.
2. Sentinel-1 patches and encoder (mode 1).
3. Campaigns: random, kriging, SHERPA mode 2.
4. Campaigns: SHERPA mode 1.
5. Figures and tables.
6. Results page.
A failed step stops the chain, except step 2 and step 4: without mode 1, the study still
runs for mode 2.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "results" / "study.log"
S1 = ROOT / "data" / "raw" / "s1_coherence"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name: str, args: list, required: bool = True) -> bool:
    log(f"start {name}: {' '.join(args)}")
    t0 = time.time()
    with open(ROOT / "results" / f"{name}.log", "w", encoding="utf-8") as out:
        code = subprocess.call([str(PY), *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                               env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    log(f"end   {name}: exit {code}, {(time.time() - t0) / 60:.0f} min")
    if code != 0 and required:
        log("the chain stops here")
        sys.exit(code)
    return code == 0


def wait_for_download(expected: int = 33, max_hours: float = 4.0) -> int:
    """Wait until the zips are complete: their count is reached and the size is stable."""
    t0, last = time.time(), -1
    while time.time() - t0 < max_hours * 3600:
        zips = list(S1.glob("*.zip"))
        size = sum(z.stat().st_size for z in zips)
        if len(zips) >= expected and size == last:
            return len(zips)
        last = size
        time.sleep(90)
    log(f"download wait ended after {max_hours} h with {len(list(S1.glob('*.zip')))} zips")
    return len(list(S1.glob("*.zip")))


if __name__ == "__main__":
    log("study start")
    n = wait_for_download()
    log(f"coherence products on disk: {n}")
    while not (ROOT / "data" / "processed" / "roads_levels.npz").exists():
        time.sleep(60)
    log("road travel times for the blockage levels: on disk")

    mode1 = run("s1_patches", ["scripts/26_s1_patches.py"], required=False)
    if mode1:
        mode1 = run("s1_encoder", ["scripts/27_s1_encoder.py", "--epochs", "40"], required=False)

    run("campaign_mode2", ["scripts/24_campaign.py", "--methods", "random,kriging,sherpa",
                           "--image", "s2_embedding_ssl4eo_ft.parquet", "--out", "campaign_mode2.parquet"])
    if mode1:
        run("campaign_mode1", ["scripts/24_campaign.py", "--methods", "sherpa",
                               "--image", "s1_embedding.parquet", "--out", "campaign_mode1.parquet"],
            required=False)
    run("campaign_figures", ["scripts/28_campaign_figures.py"])
    run("results_page", ["scripts/29_results_page.py"], required=False)
    log("study complete")
