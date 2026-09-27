# -*- coding: utf-8 -*-
"""
多模态消融实验入口:对比 影像 / 影像+临床 / 影像+临床+证型 对 anti-VEGF 耐受性预测。
用法:
  python run_multimodal.py --config configs/multimodal_antivegf.yaml
  python run_multimodal.py --config ... --mode image_clinical_syndrome   # 只跑一种
  python run_multimodal.py --config ... --smoke                          # 1epoch冒烟
"""
import os, sys, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALL_MODES = ["image", "image_clinical", "image_clinical_syndrome"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default=None, help="单独跑某模式;默认三种都跑")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        cfg["train"]["epochs"] = 1
    from tcm_retina.engines.train_multimodal import run
    modes = [args.mode] if args.mode else ALL_MODES
    results = {}
    for m in modes:
        results[m] = run(cfg, m)
    print("\n========== 消融对比(5折平均) ==========")
    for m, s in results.items():
        print(f"  {m:32s} AUC={s['auc']}  bacc={s['balanced_acc']}  sens={s['sensitivity']}")

if __name__ == "__main__":
    main()
