# -*- coding: utf-8 -*-
"""
39_setup_external_dataset.py — turn a freshly downloaded public DR dataset into the layout
that 38_external2_benchmark.py expects, without you having to rename anything by hand.

Target layout it produces:
    external_data/<name>/images/*          image files (left where they are, or moved)
    external_data/<name>/labels.csv        columns: image,grade   (grade 0-4, 4 = PDR)

SUPPORTED SOURCES
-----------------
aptos   APTOS 2019 Blindness Detection  (kaggle.com/competitions/aptos2019-blindness-detection)
        Download and unzip so that you have, anywhere on disk:
            <src>/train.csv              columns: id_code,diagnosis
            <src>/train_images/*.png
        3,662 graded images; grade 4 (proliferative) = 295. Aravind Eye Hospital, India.

idrid   IDRiD  (Grand Challenge / IEEE DataPort). Lighter download, but far fewer PDR cases,
        so the confidence interval will be wide. Expects:
            <src>/**/*.csv               a grading csv with an image-name column and a
                                         'Retinopathy grade' column
            <src>/**/*.jpg               the images (searched recursively)

generic Any dataset you have already reduced to a csv with an image column and a 0-4 grade
        column: pass --images <dir> --labels <csv> --image-col X --grade-col Y

USAGE
    python scripts/39_setup_external_dataset.py aptos --src "D:/downloads/aptos2019"
    python scripts/39_setup_external_dataset.py --help

It copies nothing by default: it writes labels.csv and creates external_data/<name>/images
as a directory junction/symlink to your download. Pass --copy to physically copy instead
(use that if the download sits on a removable drive).
"""
import os, sys, glob, argparse, shutil
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "external_data")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def link_images(src_dir, dst_dir, copy=False):
    if os.path.exists(dst_dir):
        print(f"[images] {dst_dir} already exists — leaving it alone")
        return
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    if copy:
        print(f"[images] copying {src_dir} -> {dst_dir} (this can take a while)")
        shutil.copytree(src_dir, dst_dir)
        return
    try:
        os.symlink(src_dir, dst_dir, target_is_directory=True)
        print(f"[images] linked {dst_dir} -> {src_dir}")
    except OSError:
        # Windows without developer mode: fall back to a directory junction
        rc = os.system(f'mklink /J "{dst_dir}" "{src_dir}" >nul 2>&1')
        if rc == 0:
            print(f"[images] junction {dst_dir} -> {src_dir}")
        else:
            print("[images] symlink/junction failed — copying instead")
            shutil.copytree(src_dir, dst_dir)


def write_labels(df, name, img_dir):
    """df must already have columns image,grade."""
    df = df[["image", "grade"]].copy()
    df["grade"] = df["grade"].astype(int)
    have = set(os.listdir(img_dir))
    before = len(df)
    df = df[df["image"].astype(str).isin(have)]
    out = os.path.join(EXT, name, "labels.csv")
    df.to_csv(out, index=False)
    print(f"[labels] {len(df)}/{before} rows matched files on disk -> {out}")
    print(df.groupby("grade").size().to_string())
    n4 = int((df["grade"] == 4).sum())
    n13 = int(df["grade"].isin([1, 2, 3]).sum())
    print(f"[labels] usable for the PDR-vs-NPDR proxy: {n4} PDR, {n13} NPDR "
          f"(grade 0 excluded) -> balanced subset will be {min(n4, n13) * 2} images")
    if min(n4, n13) < 100:
        print("[warn] fewer than 100 per class; the AUC confidence interval will be wide")


def do_aptos_auto(a):
    """Download APTOS through kagglehub, then run the normal aptos conversion.

    Requires, once: a Kaggle API token at ~/.kaggle/kaggle.json (Kaggle > Settings > API >
    Create New Token), and acceptance of the competition rules on the competition page.
    """
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("pip install kagglehub")
    try:
        path = kagglehub.competition_download("aptos2019-blindness-detection")
    except Exception as e:
        raise SystemExit(
            f"{type(e).__name__}: {e}\n\n"
            "If this says the user is not authenticated:\n"
            "  1. create a Kaggle account\n"
            "  2. Kaggle > Settings > API > Create New Token  (downloads kaggle.json)\n"
            "  3. move that file to  %s\n"
            "If it says 403 / rules not accepted:\n"
            "  open kaggle.com/competitions/aptos2019-blindness-detection and accept the rules"
            % os.path.expanduser("~/.kaggle/kaggle.json"))
    print(f"[kagglehub] downloaded to {path}")
    a.src = path
    return do_aptos(a)


def do_aptos(a):
    if not a.src:
        raise SystemExit("aptos mode needs --src (or use: 39_setup_external_dataset.py aptos-auto)")
    csv = os.path.join(a.src, "train.csv")
    imgs = os.path.join(a.src, "train_images")
    if not os.path.exists(csv) or not os.path.isdir(imgs):
        raise SystemExit(f"expected {csv} and {imgs}\n"
                         f"unzip the APTOS download so both exist, then rerun")
    d = pd.read_csv(csv)
    assert {"id_code", "diagnosis"} <= set(d.columns), d.columns.tolist()
    ext = os.path.splitext(os.listdir(imgs)[0])[1]
    d = d.rename(columns={"diagnosis": "grade"})
    d["image"] = d["id_code"].astype(str) + ext
    link_images(imgs, os.path.join(EXT, "aptos", "images"), a.copy)
    write_labels(d, "aptos", os.path.join(EXT, "aptos", "images"))


def do_idrid(a):
    csvs = glob.glob(os.path.join(a.src, "**", "*.csv"), recursive=True)
    rows = []
    for c in csvs:
        try:
            d = pd.read_csv(c)
        except Exception:
            continue
        gcol = next((x for x in d.columns if "retinopathy" in x.lower()), None)
        icol = next((x for x in d.columns if "image" in x.lower()), None)
        if gcol and icol:
            rows.append(pd.DataFrame({"image": d[icol].astype(str), "grade": d[gcol]}))
            print(f"[idrid] read {len(d)} rows from {os.path.basename(c)}")
    if not rows:
        raise SystemExit("no grading csv found (need an image column + a 'Retinopathy grade' column)")
    d = pd.concat(rows, ignore_index=True).dropna()

    files = glob.glob(os.path.join(a.src, "**", "*.jpg"), recursive=True)
    if not files:
        raise SystemExit("no .jpg images found under --src")
    stem2name = {os.path.splitext(os.path.basename(f))[0]: os.path.basename(f) for f in files}
    d["image"] = d["image"].map(lambda s: stem2name.get(s, s if "." in s else s + ".jpg"))

    dst = os.path.join(EXT, "idrid", "images")
    os.makedirs(dst, exist_ok=True)
    for f in files:
        t = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(t):
            shutil.copy2(f, t)
    print(f"[images] gathered {len(files)} images -> {dst}")
    write_labels(d, "idrid", dst)


def do_generic(a):
    if not (a.images and a.labels):
        raise SystemExit("generic mode needs --images and --labels")
    d = pd.read_csv(a.labels)
    d = d.rename(columns={a.image_col: "image", a.grade_col: "grade"})
    link_images(os.path.abspath(a.images), os.path.join(EXT, a.name, "images"), a.copy)
    write_labels(d, a.name, os.path.join(EXT, a.name, "images"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["aptos-auto", "aptos", "idrid", "generic"])
    ap.add_argument("--src", help="folder you unzipped the download into")
    ap.add_argument("--copy", action="store_true", help="copy images instead of linking")
    ap.add_argument("--name", default="custom", help="generic mode: folder name to create")
    ap.add_argument("--images", help="generic mode: image directory")
    ap.add_argument("--labels", help="generic mode: labels csv")
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--grade-col", default="grade")
    a = ap.parse_args()
    os.makedirs(EXT, exist_ok=True)
    {"aptos-auto": do_aptos_auto, "aptos": do_aptos,
     "idrid": do_idrid, "generic": do_generic}[a.source](a)
    nm = {"aptos-auto": "aptos", "aptos": "aptos", "idrid": "idrid",
          "generic": a.name}[a.source]
    print(f"\nnext:\n  python scripts/38_external2_benchmark.py prep --dataset {nm}")
