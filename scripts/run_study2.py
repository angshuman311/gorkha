"""Driver of the second study: four survey methods, two GORKHA modes, the 2 by 2 ablation.

It waits for the end of the first study (results/study.log), then runs:
1. Campaigns, mode 2: random, nearest, kriging_var, kriging_ivr, gorkha (estimates kept).
2. Campaigns, mode 1: gorkha.
3. The 2 by 2 ablation (estimator against selection).
4. Figures, tables, and the results page.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "results" / "study2.log"


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
    log("study 2 start")
    first = ROOT / "results" / "study.log"
    while not any(k in first.read_text(encoding="utf-8") for k in ("study complete", "the chain stops here")):
        time.sleep(120)
    log("study 1 is complete")
    run("campaign2_mode2", ["scripts/24_campaign.py", "--methods", "random,nearest,kriging_var,kriging_ivr,gorkha",
                            "--image", "s2_embedding_ssl4eo_ft.parquet", "--out", "campaign2_mode2.parquet", "--keep",
                            "--levels", "0,obs,30,60,100"])
    mode1 = (ROOT / "data" / "processed" / "s1_embedding.parquet").exists()
    if mode1:
        run("campaign2_mode1", ["scripts/24_campaign.py", "--methods", "gorkha", "--image", "s1_embedding.parquet",
                                "--out", "campaign2_mode1.parquet", "--keep", "--levels", "0,obs,30,60,100"], required=False)
    run("ablation_2x2", ["scripts/31_ablation_2x2.py"], required=False)
    run("campaign_figures", ["scripts/28_campaign_figures.py"])
    run("results_page", ["scripts/29_results_page.py"], required=False)
    log("study 2 complete")
