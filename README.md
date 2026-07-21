# ECGR 4106 Homework 5

This repository is the Homework 5 workspace for Gilberto Feliu.

Repository link for the report: <https://github.com/gfeliu-rgb/ECGR-4106-HW5>

## Files

- `src/hw5_vit_resnet.py`: Problem 1 implementation for ViT configurations and the ResNet-18 baseline.
- `src/hw5_swin.py`: Problem 2 implementation for pretrained and scratch Swin experiments.
- `tools/make_hw5_plots.py`: helper script for generating loss and accuracy figures from completed CSVs.
- `tools/run_swin_cached_extended.py`: optional extended cached-feature Swin head-training script for extra real epoch points.
- `Homework5_Problem1_ViT_vs_ResNet18.ipynb`: notebook for viewing Problem 1 metrics and plots.
- `Homework5_Problem2_Swin_Finetuning.ipynb`: notebook for viewing Problem 2 metrics and plots.
- `Results_Problem_1/`: completed Problem 1 CSV result artifacts and plots.
- `Results_Problem_2/`: completed Problem 2 CSV result artifacts and plots.
- `report/homework5_report.md`: report source covering every required section.
- `report/homework5_report.pdf`: completed PDF report for Canvas submission.
- `SUBMISSION_CHECKLIST.md`: final packaging checklist before Canvas submission.

## Reproduce Final Artifacts

```bash
python src/hw5_vit_resnet.py --run-all-vit --run-resnet
python src/hw5_swin.py --run-pretrained swin_tiny_pretrained
python src/hw5_swin.py --run-pretrained swin_small_pretrained
python src/hw5_swin.py --run-scratch --epochs 5 --batch-size 32 --image-size 224
python tools/make_hw5_plots.py
```

## Notes

- The structure is aligned with the course `Homework_5` repository layout.
- This repository contains the completed Homework 5 source, notebooks, result CSVs, generated plots, and PDF report.
- Both Jupyter notebooks are executed in place with full source-code sections split into step-by-step output blocks, visible tables, solid-line training-curve figures, tradeoff figures, and markdown interpretation.
- Problem 1 includes completed 10-epoch results for all planned models.
- Problem 2 includes completed 5-epoch CUDA results for pretrained Swin-Tiny, pretrained Swin-Small, and scratch Swin.
- Problem 2 also includes a clearly labeled extended cached-head analysis with 100 real measured epochs for pretrained Swin-Tiny and Swin-Small on a balanced CIFAR-100 subset.
