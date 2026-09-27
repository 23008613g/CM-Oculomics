# -*- coding: utf-8 -*-
"""多任务消融:对比 单任务(aux=0) vs 多任务(证型辅助监督) 对主任务(耐受性)的影响。
用法: python run_multitask.py --config configs/multitask_antivegf.yaml
      python run_multitask.py --config ... --smoke
"""
import os, sys, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aux", type=float, default=None, help="只跑某个aux_weight")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f: cfg = yaml.safe_load(f)
    if args.smoke: cfg["train"]["epochs"] = 1
    from tcm_retina.engines.train_multitask import run
    runs = [(args.aux, f"aux{args.aux}")] if args.aux is not None else \
           [(0.0, "single"), (0.3, "multitask")]
    results = {}
    for w, tag in runs:
        results[tag] = run(cfg, w, tag)
    print("\n========== 多任务消融对比(5折平均) ==========")
    for tag, s in results.items():
        print(f"  {tag:12s} AUC={s['auc']}  bacc={s['balanced_acc']}  sens={s['sensitivity']}")

if __name__ == "__main__":
    main()
