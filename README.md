# ECGR 5106 Homework 5

This repository is the Homework 5 workspace for Gilberto Feliu.

Repository link for the report: <https://github.com/gfeliu-rgb/ECGR-5106-HW5>

## Files

- `src/hw5_vit_resnet.py`: Problem 1 implementation for ViT configurations and the ResNet-18 baseline.
- `src/hw5_swin.py`: Problem 2 implementation for pretrained and scratch Swin experiments.
- `tools/make_hw5_plots.py`: helper script for generating loss-curve figures from exported CSVs.
- `Homework5_Problem1_ViT_vs_ResNet18.ipynb`: notebook for viewing Problem 1 metrics and plots.
- `Homework5_Problem2_Swin_Finetuning.ipynb`: notebook for viewing Problem 2 metrics and plots.
- `Results_Problem_1/`: Problem 1 CSV templates and output artifacts.
- `Results_Problem_2/`: Problem 2 CSV templates and output artifacts.
- `report/homework5_report.md`: report source covering every required section.
- `SUBMISSION_CHECKLIST.md`: final packaging checklist before Canvas submission.

## Reproduce Templates

```bash
python src/hw5_vit_resnet.py --export-templates
python src/hw5_swin.py --export-templates
python tools/make_hw5_plots.py
```

## Notes

- The structure is aligned with the course `Homework_5` repository layout.
- Only reusable scaffold elements were mirrored here. No third-party homework solutions were copied.
- Problem 1 includes completed 10-epoch results for all planned models.
- Problem 2 includes completed 5-epoch CUDA results for pretrained Swin-Tiny, pretrained Swin-Small, and scratch Swin.
