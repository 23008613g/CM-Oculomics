# -*- coding: utf-8 -*-
"""
55_ddr_deploy_score.py — score the DINOv2 deploy model on the DDR subset.

WHY: results/dino_experiment/ddr_dinov2_l_res224.npz is the single-model external DDR
result that Section 2.3 and Figure 5b report (as distinct from the seed ensemble). It was
produced by `27_dino_experiment.py standard` back in June, before the overlay masking, and
nothing in the retraining queue rewrites it. Re-running the whole `standard` pipeline just
for this one file would repeat a learning-rate sweep and a full cross-validation, so this
script loads the deploy model that 30_dino_artifacts.py just trained and scores DDR with the
same helper the original used.

USAGE   python scripts/55_ddr_deploy_score.py
OUTPUT  results/dino_experiment/ddr_dinov2_l_res224.npz   (backup: *_preMask.npz)
"""
import os, sys, json, shutil, importlib.util
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


EXP = _load(os.path.join(ROOT, "scripts", "27_dino_experiment.py"), "exp27")
_ORIG = EXP.DataLoader                       # Windows spawn cannot pickle a dynamic module
EXP.DataLoader = lambda *a, **k: _ORIG(*a, **{**k, "num_workers": 0})

OUT = os.path.join(ROOT, "results", "dino_experiment")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES = 224

# Both deploy models are trained on fold != 0 with early stopping on fold 0, so the two DDR
# scores in Figure 3c come from matched single models rather than from different protocols.
JOBS = [
    ("dinov2_l", os.path.join(OUT, "dino_deploy.pth"),
     os.path.join(OUT, "ddr_dinov2_l_res224.npz"), "dino_deploy.pth (30_dino_artifacts.py)"),
    ("retfound", os.path.join(ROOT, "results", "retfound_antivegf", "best.pth"),
     os.path.join(ROOT, "results", "external_valpred.npz"),
     "retfound_antivegf/best.pth (run.py train, masked images)"),
]


def main():
    summary = {}
    for kind, deploy, dst, src in JOBS:
        if not os.path.exists(deploy):
            print("SKIP %s: %s not found" % (kind, deploy)); continue
        bak = dst.replace(".npz", "_preMask.npz")
        if os.path.exists(dst) and not os.path.exists(bak):
            shutil.copy2(dst, bak); print("backed up -> %s" % os.path.basename(bak))

        model = EXP.make_model(kind, RES).to(DEVICE)
        model.load_state_dict(torch.load(deploy, map_location=DEVICE))
        print("loaded %s for %s" % (os.path.basename(deploy), kind))

        ys, ps = EXP.eval_ddr(model, RES)
        auc = float(roc_auc_score(ys, ps)); ap = float(average_precision_score(ys, ps))
        np.savez(dst, ys=ys, pos=ps)
        print("  DDR subset n=%d (%d PDR):  AUC = %.4f   AP = %.4f"
              % (len(ys), int(ys.sum()), auc, ap))
        print("  saved %s" % os.path.basename(dst))
        summary[kind] = {"ddr_auc": round(auc, 4), "ddr_ap": round(ap, 4),
                         "n": int(len(ys)), "n_pdr": int(ys.sum()), "source": src}
        del model
        torch.cuda.empty_cache()
    json.dump(summary, open(os.path.join(OUT, "ddr_deploy_score.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
