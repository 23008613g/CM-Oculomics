# -*- coding: utf-8 -*-
"""
38_external2_benchmark.py — replicate the generalist-vs-specialist external comparison on a
SECOND, independent public dataset.

WHY: the manuscript's headline claim (a generalist vision foundation model transfers better
out of distribution than a retina-specialist one) currently rests on ONE external dataset
(DDR, 550 images) evaluated on a PDR-vs-NPDR proxy task. A reviewer will ask whether a
single external set supports a claim about pretraining strategy. This script answers that by
running the identical protocol on a second dataset, across several seeds.

PROTOCOL (identical to the published one, scripts/27_dino_experiment.py):
  * train on the in-house cohort, folds != 0, early-stop on fold 0
  * matched input resolution 224, effective batch 16, published per-backbone LR
      dinov2_l  lr 1e-5      retfound  lr 5e-5
  * score external sets with ZERO fine-tuning
  * external label mapping, exactly as for DDR: grade 4 = PDR (positive),
    grades 1-3 = NPDR (negative), grade 0 (no DR) EXCLUDED, class-balanced subset

USAGE
  1) prepare the second dataset (see --help of `prep` for the expected raw layout):
       python scripts/38_external2_benchmark.py prep --dataset aptos
  2) train + score (this is the long step; ~20-30 min per model on an RTX 3090):
       python scripts/38_external2_benchmark.py run --dataset aptos --seeds 3
  3) aggregate, DeLong, bootstrap CIs, figure:
       python scripts/38_external2_benchmark.py report --dataset aptos

OUTPUTS  results/external2/
  subset_labels_<ds>.csv            the balanced subset actually scored
  pred_<backbone>_seed<k>_<set>.npz  per-model predictions on each external set
  external2_summary.json             all AUCs, DeLong p-values, bootstrap CIs
  Figure_external2.png               ROC panel + per-seed AUC comparison
"""
import os, sys, json, glob, random, argparse, importlib.util, time
import numpy as np, pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, balanced_accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from tcm_retina.data.dataset import build_transforms

EXT = os.path.join(ROOT, "external_data")
OUT = os.path.join(ROOT, "results", "external2")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = {"dinov2_l": 1e-5, "retfound": 5e-5}     # published per-backbone learning rates
RES = 224
PUBLISHED_DDR = {"dinov2_l": 0.921, "retfound": 0.858}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EXP = _load(os.path.join(ROOT, "scripts", "27_dino_experiment.py"), "exp27")
DL = _load(os.path.join(ROOT, "scripts", "06_delong.py"), "delong06")

# 27_dino_experiment.py builds its DataLoaders with num_workers=4. Because we import it
# dynamically (its filename starts with a digit, so it is not importable by name), Windows
# spawn-based workers cannot re-import the module and fail to unpickle its Dataset class.
# Force single-process loading; the GPU forward/backward dominates for a cohort this size.
_ORIG_DATALOADER = EXP.DataLoader


def _dataloader_no_workers(*a, **kw):
    kw["num_workers"] = 0
    return _ORIG_DATALOADER(*a, **kw)


EXP.DataLoader = _dataloader_no_workers


# --------------------------------------------------------------------------- data
class ExternalDS(Dataset):
    """Reads <root>/images/<image> with a `grade` column; positive class = grade 4 (PDR)."""

    def __init__(self, lab, img_dir, res):
        self.lab = lab.reset_index(drop=True)
        self.img_dir = img_dir
        self.tf = build_transforms(res, False, None)

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        r = self.lab.iloc[i]
        x = self.tf(Image.open(os.path.join(self.img_dir, str(r["image"]))).convert("RGB"))
        return x, int(r["grade"] == 4), i


def dataset_paths(ds):
    """DDR keeps its original layout; every other dataset uses the generic one."""
    if ds == "ddr":
        d = os.path.join(EXT, "ddr")
        lab = pd.read_csv(os.path.join(d, "subset_labels.csv")).rename(
            columns={"ddr_grade": "grade"})
        return lab[["image", "grade"]], os.path.join(d, "images")
    d = os.path.join(EXT, ds)
    sub = os.path.join(OUT, f"subset_labels_{ds}.csv")
    if not os.path.exists(sub):
        raise SystemExit(f"missing {sub} — run:  python {sys.argv[0]} prep --dataset {ds}")
    return pd.read_csv(sub)[["image", "grade"]], os.path.join(d, "images")


def cmd_prep(args):
    """Build the balanced PDR-vs-NPDR subset, applying the DDR rule exactly.

    Expects: external_data/<ds>/images/*        the image files
             external_data/<ds>/labels.csv      columns: image,grade   (grade 0-4)
    """
    d = os.path.join(EXT, args.dataset)
    raw = os.path.join(d, "labels.csv")
    if not os.path.exists(raw):
        raise SystemExit(
            f"missing {raw}\n"
            f"Expected layout:\n"
            f"  {d}/images/*.png|jpg\n"
            f"  {d}/labels.csv   with columns  image,grade   (grade 0-4, 4 = proliferative)")
    lab = pd.read_csv(raw)
    lab.columns = [c.strip().lower() for c in lab.columns]
    assert {"image", "grade"} <= set(lab.columns), f"labels.csv needs image,grade; got {list(lab.columns)}"
    lab["grade"] = lab["grade"].astype(int)

    have = set(os.listdir(os.path.join(d, "images")))
    lab = lab[lab["image"].astype(str).isin(have)]
    print(f"[prep] {len(lab)} labelled images found on disk")
    print(lab.groupby("grade").size().to_string())

    pos = lab[lab["grade"] == 4]
    neg = lab[lab["grade"].isin([1, 2, 3])]            # grade 0 excluded, exactly as for DDR
    n = min(len(pos), len(neg))
    rng = np.random.RandomState(42)
    pos_s = pos.sample(n, random_state=rng) if len(pos) > n else pos
    neg_s = neg.sample(n, random_state=rng) if len(neg) > n else neg
    sub = pd.concat([pos_s, neg_s]).sample(frac=1, random_state=42).reset_index(drop=True)
    sub["label"] = np.where(sub["grade"] == 4, "PDR", "NPDR")
    fn = os.path.join(OUT, f"subset_labels_{args.dataset}.csv")
    sub.to_csv(fn, index=False)
    print(f"[prep] balanced subset: {len(sub)} images ({n} PDR / {n} NPDR)  -> {fn}")
    print(sub.groupby(['grade', 'label']).size().to_string())


# --------------------------------------------------------------------------- scoring
@torch.no_grad()
def score(model, lab, img_dir, res=RES, bs=16):
    ld = DataLoader(ExternalDS(lab, img_dir, res), bs, shuffle=False, num_workers=0)
    ys = np.zeros(len(lab)); ps = np.zeros(len(lab))
    model.eval()
    for x, y, idx in ld:
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            p = torch.softmax(model(x.to(DEVICE)), 1)[:, 1].float().cpu().numpy()
        ps[idx.numpy()] = p; ys[idx.numpy()] = y.numpy()
    return ys, ps


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def cmd_run(args):
    sets = ["ddr"] + list(args.dataset)
    loaded = {s: dataset_paths(s) for s in sets}
    for s, (lab, _) in loaded.items():
        print(f"[data] {s}: {len(lab)} images, PDR={int((lab['grade'] == 4).sum())}")

    tv = EXP.load_tv()
    tr, va = tv[tv.fold != 0], tv[tv.fold == 0]
    print(f"[data] in-house train={len(tr)} earlystop-fold0={len(va)}")

    for kind in args.backbones:
        for seed in range(args.seeds):
            tag = f"{kind}_seed{seed}"
            ckpt = os.path.join(OUT, f"ckpt_{tag}.pth")
            todo = [s for s in sets
                    if args.force or not os.path.exists(os.path.join(OUT, f"pred_{tag}_{s}.npz"))]
            if not todo:
                print(f"[skip] {tag} already scored on every requested set")
                continue
            t0 = time.time()
            set_seed(seed)
            if os.path.exists(ckpt) and not args.retrain:
                # reuse the exact model from the earlier run: no retraining needed to add a set
                model = EXP.make_model(kind, RES).to(DEVICE)
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                print(f"[load ] {tag}  <- {os.path.basename(ckpt)}", flush=True)
            else:
                print(f"[train] {tag}  lr={LR[kind]:.0e}  res={RES} ...", flush=True)
                _bp, _pk, model = EXP.train_fold(tr, va, kind, RES, LR[kind],
                                                 args.bs, return_model=True)
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)
                print(f"        saved {os.path.basename(ckpt)}", flush=True)
            for s in todo:
                lab, img_dir = loaded[s]
                ys, ps = score(model, lab, img_dir)
                np.savez(os.path.join(OUT, f"pred_{tag}_{s}.npz"), ys=ys, pos=ps)
                print(f"        {s:>10}  AUC={roc_auc_score(ys, ps):.4f}", flush=True)
            del model; torch.cuda.empty_cache()
            print(f"        done in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("\nall runs finished — now:  python %s report --dataset %s"
          % (sys.argv[0], " ".join(args.dataset)))


# --------------------------------------------------------------------------- report
def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.RandomState(seed); a = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        a.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def cmd_report(args):
    sets = ["ddr"] + list(args.dataset)
    res = {"protocol": {"resolution": RES, "lr": LR, "seeds": args.seeds,
                        "label_rule": "PDR = grade 4; NPDR = grades 1-3; grade 0 excluded",
                        "fine_tuning_on_external": "none"},
           "published_ddr": PUBLISHED_DDR, "per_seed": {}, "ensemble": {}}

    for s in sets:
        res["per_seed"][s] = {}
        for kind in args.backbones:
            aucs, probs, ys = [], [], None
            for seed in range(args.seeds):
                f = os.path.join(OUT, f"pred_{kind}_seed{seed}_{s}.npz")
                if not os.path.exists(f):
                    continue
                z = np.load(f); ys = z["ys"]; probs.append(z["pos"])
                aucs.append(float(roc_auc_score(z["ys"], z["pos"])))
            if not aucs:
                continue
            res["per_seed"][s][kind] = {"aucs": aucs,
                                        "mean": float(np.mean(aucs)),
                                        "min": float(np.min(aucs)),
                                        "max": float(np.max(aucs))}
            res.setdefault("_arrays", {})[f"{s}|{kind}"] = (ys, np.mean(probs, 0))

    # ensemble (mean probability across seeds) + DeLong + bootstrap CI
    for s in sets:
        k1, k2 = "dinov2_l", "retfound"
        if f"{s}|{k1}" not in res.get("_arrays", {}) or f"{s}|{k2}" not in res["_arrays"]:
            continue
        y, p1 = res["_arrays"][f"{s}|{k1}"]
        _, p2 = res["_arrays"][f"{s}|{k2}"]
        a1, a2 = roc_auc_score(y, p1), roc_auc_score(y, p2)
        try:
            _a1, _a2, _z, pval = DL.delong_test(y.astype(int), p1, p2)   # returns auc1,auc2,z,p
        except Exception as e:
            pval = float("nan")
            print(f"[warn] DeLong failed on {s}: {e}")
        res["ensemble"][s] = {
            "n": int(len(y)), "n_pos": int(y.sum()),
            "dinov2_auc": float(a1), "dinov2_ci": boot_ci(y, p1),
            "retfound_auc": float(a2), "retfound_ci": boot_ci(y, p2),
            "delta_auc": float(a1 - a2), "delong_p": float(pval),
            "dinov2_ap": float(average_precision_score(y, p1)),
            "retfound_ap": float(average_precision_score(y, p2)),
        }

    arrays = res.pop("_arrays", {})
    json.dump(res, open(os.path.join(OUT, "external2_summary.json"), "w"), indent=2)

    # ---- console table ----
    print("\n" + "=" * 74)
    print("GENERALIST vs RETINA-SPECIALIST — external transfer, zero fine-tuning")
    print("=" * 74)
    for s in sets:
        e = res["ensemble"].get(s)
        if not e:
            continue
        star = "  <-- published: DINOv2 %.3f / RETFound %.3f" % (
            PUBLISHED_DDR["dinov2_l"], PUBLISHED_DDR["retfound"]) if s == "ddr" else ""
        print(f"\n{s.upper()}  (n={e['n']}, PDR={e['n_pos']}){star}")
        print(f"   DINOv2    AUC {e['dinov2_auc']:.4f}  95%CI [{e['dinov2_ci'][0]:.3f}, {e['dinov2_ci'][1]:.3f}]")
        print(f"   RETFound  AUC {e['retfound_auc']:.4f}  95%CI [{e['retfound_ci'][0]:.3f}, {e['retfound_ci'][1]:.3f}]")
        print(f"   dAUC {e['delta_auc']:+.4f}   DeLong p = {e['delong_p']:.2e}")
        for kind in args.backbones:
            ps_ = res["per_seed"][s].get(kind)
            if ps_:
                print(f"   per-seed {kind:9s}: " + ", ".join(f"{a:.4f}" for a in ps_["aucs"])
                      + f"   (mean {ps_['mean']:.4f}, range {ps_['max'] - ps_['min']:.4f})")

    _figure(res, arrays, sets, args)
    print(f"\nsaved {os.path.join(OUT, 'external2_summary.json')}")


def _figure(res, arrays, sets, args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"dinov2_l": "#D55E00", "retfound": "#0072B2"}
    plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(sets) + 1, figsize=(5.2 * (len(sets) + 1), 4.6))
    for ax, s in zip(axes, sets):
        for kind in args.backbones:
            k = f"{s}|{kind}"
            if k not in arrays:
                continue
            y, p = arrays[k]
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=C[kind], lw=2.4,
                    label=f"{'DINOv2' if kind == 'dinov2_l' else 'RETFound'}  {roc_auc_score(y, p):.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        e = res["ensemble"].get(s, {})
        ax.set_title(f"{s.upper()}  (n={e.get('n', '?')})\nDeLong p = {e.get('delong_p', float('nan')):.1e}")
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
        ax.legend(loc="lower right", title="AUC")
    # per-seed dot plot
    ax = axes[-1]
    xt, xl = [], []
    for i, s in enumerate(sets):
        for j, kind in enumerate(args.backbones):
            d = res["per_seed"][s].get(kind)
            if not d:
                continue
            x = i * 2.4 + j
            ax.scatter([x] * len(d["aucs"]), d["aucs"], s=70, color=C[kind], zorder=3)
            ax.plot([x - 0.22, x + 0.22], [d["mean"]] * 2, color="k", lw=2, zorder=4)
            xt.append(x); xl.append(f"{'DINOv2' if kind == 'dinov2_l' else 'RETF'}\n{s}")
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=9)
    ax.set_ylabel("External AUC"); ax.set_title("Per-seed reproducibility\n(bar = mean)")
    fig.tight_layout()
    fn = os.path.join(OUT, "Figure_external2.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close()
    print(f"saved {fn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "run", "report"):
        s = sub.add_parser(name)
        if name == "prep":
            s.add_argument("--dataset", required=True, help="folder name under external_data/")
        else:
            s.add_argument("--dataset", required=True, nargs="+",
                           help="one or more folder names under external_data/ (ddr is always added)")
            s.add_argument("--seeds", type=int, default=3)
            s.add_argument("--backbones", nargs="+", default=["dinov2_l", "retfound"])
        if name == "run":
            s.add_argument("--bs", type=int, default=16, help="per-step batch size")
            s.add_argument("--force", action="store_true",
                           help="re-score even if predictions already exist")
            s.add_argument("--retrain", action="store_true",
                           help="ignore saved checkpoints and train from scratch")
    a = ap.parse_args()
    {"prep": cmd_prep, "run": cmd_run, "report": cmd_report}[a.cmd](a)
