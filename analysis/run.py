# -*- coding: utf-8 -*-
"""
TCM-Retina CLI 入口
===================
用法:
  python run.py train --config configs/baseline_syndrome.yaml
  python run.py train --config configs/baseline_syndrome.yaml --fold 1 --epochs 5
  python run.py cv    --config configs/baseline_syndrome.yaml      # 跑5折
  python run.py smoke --config configs/baseline_syndrome.yaml      # 1 epoch 冒烟测试

设计为可复现 pipeline:所有超参在 YAML;命令行可覆盖少量常用项。
"""
import os, sys, argparse, json
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_cfg(path, overrides):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 命令行覆盖
    if overrides.get("fold") is not None:
        cfg["train"]["fold"] = overrides["fold"]
    if overrides.get("epochs") is not None:
        cfg["train"]["epochs"] = overrides["epochs"]
    if overrides.get("batch_size") is not None:
        cfg["train"]["batch_size"] = overrides["batch_size"]
    return cfg


def main():
    ap = argparse.ArgumentParser(description="TCM-Retina pipeline")
    ap.add_argument("command", choices=["train", "cv", "smoke"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    from tcm_retina.engines.train_classify import run

    if args.command == "smoke":
        cfg = load_cfg(args.config, {"epochs": 1, "fold": args.fold or 0,
                                     "batch_size": args.batch_size})
        cfg["output"]["dir"] = cfg["output"]["dir"] + "_smoke"
        run(cfg)
    elif args.command == "train":
        cfg = load_cfg(args.config, vars(args))
        run(cfg)
    elif args.command == "cv":
        base = load_cfg(args.config, {"epochs": args.epochs})
        all_res = []
        for k in range(5):
            print(f"\n{'='*60}\n FOLD {k}\n{'='*60}")
            base["train"]["fold"] = k
            all_res.append(run(dict(base)))
        # 汇总
        import numpy as np
        keys = ["accuracy", "balanced_acc", "macro_f1", "macro_auc", "kappa"]
        summary = {k: f"{np.mean([r[k] for r in all_res]):.4f} ± "
                      f"{np.std([r[k] for r in all_res]):.4f}" for k in keys}
        out = os.path.join(base["output"]["dir"], "cv_summary.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n========== 5-Fold CV 汇总 ==========")
        for k, v in summary.items():
            print(f"  {k:14s}: {v}")
        print("已保存:", out)


if __name__ == "__main__":
    main()
