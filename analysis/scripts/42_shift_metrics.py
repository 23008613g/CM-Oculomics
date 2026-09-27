# -*- coding: utf-8 -*-
"""
42_shift_metrics.py — can the transfer loss be anticipated BEFORE labels exist?

WHY: the seven-cohort analysis establishes that external AUC tracks the cohort rather than
the backbone, but a reviewer's next question is operational: given a new cohort, what should
a deployer do? This script tests whether a label-free measure of representation shift,
computable from unlabelled target images alone, predicts the AUC that will be observed once
labels arrive.

For each fine-tuned model we embed the development cohort and each external cohort at the
encoder's penultimate layer, reduce to a common low-dimensional space (PCA fitted on the
development cohort only), and compute three standard divergence measures:

  frechet   Frechet distance between Gaussian fits          (mean + covariance shift)
  mmd2      squared maximum mean discrepancy, RBF kernel     (distribution-free)
  dom_auc   cross-validated AUC of a logistic classifier separating development from target
            images ("proxy A-distance"): 0.5 = indistinguishable, 1.0 = trivially separable

Each is then correlated (Spearman, n = 7 cohorts) against the external AUC that the same
model actually achieved. None of the three uses target labels at any point.

USAGE
  python scripts/42_shift_metrics.py                 # all backbones, all seeds
  python scripts/42_shift_metrics.py --seeds 1       # quick look

OUTPUTS  results/external2/
  shift_metrics.json           per-cohort metrics, per backbone, plus the correlations
  Figure_shift.png             (a) shift per cohort  (b) shift versus external AUC
"""
import os, sys, json, argparse, importlib.util
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import linalg, stats
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = os.path.join(ROOT, "results", "external2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES = 224
NPC = 16          # PCA dimension: comparisons run at n=70, so keep n well above d


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EXP = _load(os.path.join(ROOT, "scripts", "27_dino_experiment.py"), "exp27")
B38 = _load(os.path.join(ROOT, "scripts", "38_external2_benchmark.py"), "ext38")

SETS = [("ddr", "DDR (China)"), ("jsiec", "JSIEC (China)"), ("deepdrid", "DeepDRiD (China)"),
        ("aptos", "APTOS (India)"), ("idrid", "IDRiD (India)"),
        ("messidor2", "Messidor-2 (France)"), ("eyepacs", "EyePACS (USA)")]
CHINA = {"ddr", "jsiec", "deepdrid"}
C = {"dinov2_l": "#D55E00", "retfound": "#0072B2"}


EMB = os.path.join(OUT, "emb")
os.makedirs(EMB, exist_ok=True)


def encoder(model):
    """The feature extractor, whichever of the two architectures this is."""
    return model[0] if isinstance(model, nn.Sequential) else model.backbone


def cached(model, ds, tag):
    """Embeddings are expensive and deterministic, so compute each one once."""
    fn = os.path.join(EMB, tag + ".npy")
    if os.path.exists(fn):
        return np.load(fn)
    f = embed(model, ds)
    np.save(fn, f)
    return f


@torch.no_grad()
def embed(model, ds, bs=32):
    """Penultimate-layer features, in dataset order."""
    enc = encoder(model).eval()
    ld = DataLoader(ds, bs, shuffle=False, num_workers=0)
    out = None
    for x, _y, idx in ld:
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            f = enc(x.to(DEVICE)).float().cpu().numpy()
        if out is None:
            out = np.zeros((len(ds), f.shape[1]), dtype=np.float32)
        out[idx.numpy()] = f
    return out


# ------------------------------------------------------------------ divergence measures
def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cm, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cm))


def mmd2_rbf(a, b):
    """Unbiased MMD^2, bandwidth by the median heuristic on the pooled sample."""
    z = np.vstack([a, b])
    d2 = distance.squareform(distance.pdist(z, "sqeuclidean"))
    med = np.median(d2[d2 > 0])
    k = np.exp(-d2 / med)
    n, m = len(a), len(b)
    kxx = (k[:n, :n].sum() - np.trace(k[:n, :n])) / (n * (n - 1))
    kyy = (k[n:, n:].sum() - np.trace(k[n:, n:])) / (m * (m - 1))
    kxy = k[:n, n:].mean()
    return float(kxx + kyy - 2 * kxy)


def domain_auc(a, b, seed=0):
    """Cross-validated separability of the two cohorts: the proxy A-distance."""
    x = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in cv.split(x, y):
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(x[tr], y[tr])
        p[te] = lr.predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y, p))


def size_matched(zdev, zext, n, reps=20, seed=0):
    """All three measures are sample-size sensitive and the cohorts differ 20-fold in size,
    so every comparison is made at a common n, averaged over repeated draws."""
    rng = np.random.RandomState(seed)
    acc = {k: [] for k in ("frechet", "mmd2", "dom_auc")}
    for r in range(reps):
        a = zdev[rng.choice(len(zdev), n, replace=False)]
        b = zext[rng.choice(len(zext), n, replace=False)]
        acc["frechet"].append(frechet(a, b))
        acc["mmd2"].append(mmd2_rbf(a, b))
        acc["dom_auc"].append(domain_auc(a, b, seed=r))
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ------------------------------------------------------------------ main
def main(args):
    meta = json.load(open(os.path.join(OUT, "meta_summary.json")))
    obs = {r["key"]: {"dinov2_l": r["dino"], "retfound": r["ret"]} for r in meta["rows"]}

    tv = EXP.load_tv()
    dev_ds = EXP.DS(tv, False, RES)
    ext_ds = {}
    for s, _ in SETS:
        lab, img_dir = B38.dataset_paths(s)
        ext_ds[s] = B38.ExternalDS(lab, img_dir, RES)
    print(f"[data] development cohort {len(tv)} images; "
          + ", ".join(f"{s}={len(ext_ds[s])}" for s, _ in SETS), flush=True)

    res = {}
    for kind in args.backbones:
        per_seed = {s: [] for s, _ in SETS}
        for seed in range(args.seeds):
            ck = os.path.join(OUT, f"ckpt_{kind}_seed{seed}.pth")
            if not os.path.exists(ck):
                print(f"[warn] missing {os.path.basename(ck)} — skipped")
                continue
            model = EXP.make_model(kind, RES).to(DEVICE)
            model.load_state_dict(torch.load(ck, map_location=DEVICE))
            print(f"[embed] {kind} seed{seed}", flush=True)

            fdev = cached(model, dev_ds, f"{kind}_seed{seed}_DEV")
            pca = PCA(NPC, random_state=0).fit(fdev)          # fitted on development only
            zdev = pca.transform(fdev)
            zext = {s: pca.transform(cached(model, ext_ds[s], f"{kind}_seed{seed}_{s}"))
                    for s, _ in SETS}
            del model
            torch.cuda.empty_cache()

            nmin = min(len(z) for z in zext.values())          # the smallest cohort sets n
            for s, _ in SETS:
                m = size_matched(zdev, zext[s], nmin, reps=args.reps, seed=seed)
                per_seed[s].append(m)
                print(f"         {s:>10}  FD={m['frechet']:8.2f}  "
                      f"MMD2={m['mmd2']:.4f}  domAUC={m['dom_auc']:.3f}", flush=True)

        rows = []
        for s, label in SETS:
            if not per_seed[s]:
                continue
            rows.append(dict(key=s, label=label, n=len(ext_ds[s]),
                             external_auc=obs[s][kind],
                             **{k: float(np.mean([d[k] for d in per_seed[s]]))
                                for k in ("frechet", "mmd2", "dom_auc")},
                             **{k + "_sd": float(np.std([d[k] for d in per_seed[s]], ddof=1))
                                if len(per_seed[s]) > 1 else 0.0
                                for k in ("frechet", "mmd2", "dom_auc")}))
        corr = {}
        for k in ("frechet", "mmd2", "dom_auc"):
            rho, p = stats.spearmanr([r[k] for r in rows], [r["external_auc"] for r in rows])
            corr[k] = dict(spearman_rho=float(rho), p=float(p), n=len(rows))
        res[kind] = dict(rows=rows, correlations=corr)

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 78)
    print("LABEL-FREE SHIFT VERSUS OBSERVED EXTERNAL AUC")
    print("=" * 78)
    for kind, r in res.items():
        print(f"\n{kind}")
        print(f"{'cohort':<22}{'n':>6}{'extAUC':>9}{'Frechet':>11}{'MMD2':>9}{'domAUC':>9}")
        for x in r["rows"]:
            print(f"{x['label']:<22}{x['n']:>6}{x['external_auc']:>9.3f}"
                  f"{x['frechet']:>11.2f}{x['mmd2']:>9.4f}{x['dom_auc']:>9.3f}")
        for k, c in r["correlations"].items():
            print(f"  Spearman {k:<9} vs external AUC: rho={c['spearman_rho']:+.3f} "
                  f"p={c['p']:.3f} (n={c['n']})")

    json.dump(res, open(os.path.join(OUT, "shift_metrics.json"), "w"), indent=2)

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    prim = args.backbones[0]
    rows = res[prim]["rows"]

    a = ax[0]
    x = np.arange(len(rows))
    a.bar(x, [r["dom_auc"] for r in rows], 0.6,
          color=["#F0E442" if r["key"] in CHINA else "#999999" for r in rows],
          edgecolor="k", linewidth=0.6)
    a.axhline(0.5, color="k", ls=":", lw=1.2)
    a.text(len(rows) - 0.4, 0.515, "indistinguishable", ha="right", fontsize=8)
    a.set_xticks(x)
    a.set_xticklabels([r["key"].upper() for r in rows], fontsize=9, rotation=20,
                      ha="right", rotation_mode="anchor")
    a.set_ylim(0.45, 1.03)
    a.set_ylabel("domain-classifier AUC\n(development vs target)")
    a.set_title("(a) Label-free representation shift\nyellow = same population as development",
                fontsize=10)

    a = ax[1]
    for kind in args.backbones:
        rr = res[kind]["rows"]
        a.scatter([r["dom_auc"] for r in rr], [r["external_auc"] for r in rr],
                  s=34 + 90 * np.array([r["n"] for r in rr]) / max(r["n"] for r in rr),
                  color=C.get(kind, "#444"), label=kind, alpha=0.85, zorder=3)
    for r in res[prim]["rows"]:
        a.annotate(r["key"].upper(), (r["dom_auc"], r["external_auc"]),
                   textcoords="offset points", xytext=(6, -3), fontsize=7.5, color="#444")
    a.set_xlabel("domain-classifier AUC  (higher = larger shift, no labels used)")
    a.set_ylabel("observed external AUC")
    c = res[prim]["correlations"]["dom_auc"]
    a.set_title(f"(b) Shift anticipates transfer loss\nSpearman $\\rho$={c['spearman_rho']:+.2f}, "
                f"p={c['p']:.3f} ({prim}, n={c['n']})", fontsize=10)
    a.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    fn = os.path.join(OUT, "Figure_shift.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {fn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20,
                    help="size-matched resampling draws per cohort")
    ap.add_argument("--backbones", nargs="+", default=["dinov2_l", "retfound"])
    main(ap.parse_args())
