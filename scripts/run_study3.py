"""Driver of the third study: the four survey methods on the roads and trails of 2015, with
the landslide blockage calibrated to the observed access (60%), and the two GORKHA modes.

It waits for the end of the second study (results/study2.log).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "results" / "study3.log"


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


if __name__ == "__main__":
    log("study 3 start")
    second = ROOT / "results" / "study2.log"
    while not (second.exists() and any(k in second.read_text(encoding="utf-8")
                                       for k in ("study 2 complete", "the chain stops here"))):
        time.sleep(120)
    while not (ROOT / "data" / "processed" / "roads_2015_levels.npz").exists():
        time.sleep(60)
    log("study 2 is complete, the 2015 travel times are on disk")
    common = ["--roads", "roads_2015_levels.npz", "--levels", "0,60,100", "--keep"]
    run("campaign3_mode2", ["scripts/24_campaign.py", "--methods", "random,nearest,kriging_var,kriging_ivr,gorkha",
                            "--image", "s2_embedding_ssl4eo_ft.parquet", "--out", "campaign3_mode2.parquet", *common])
    if (ROOT / "data" / "processed" / "s1_embedding.parquet").exists():
        run("campaign3_mode1", ["scripts/24_campaign.py", "--methods", "gorkha", "--image", "s1_embedding.parquet",
                                "--out", "campaign3_mode1.parquet", *common], required=False)
    run("ablation_2x2_2015", ["scripts/31_ablation_2x2.py", "--file", "campaign3_mode2.parquet", "--levels", "0,60"],
        required=False)
    run("campaign_figures_2015", ["scripts/28_campaign_figures.py", "--prefix", "campaign3", "--suffix", "_2015"])
    run("results_page_2015", ["scripts/29_results_page.py", "--suffix", "_2015", "--roads",
                              "roads and trails of 2015, Survey Department layer on HDX", "--ablation", "ablation_2x2_2015.csv"],
        required=False)
    log("study 3 complete")
