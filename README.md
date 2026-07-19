# ECGR 5106 Homework 5

This repository is the Homework 5 workspace for Gilberto Feliu.

Repository link for the report: `TBD`

## Files

- `src/hw5_vit_resnet.py`: Problem 1 scaffold for ViT configurations and the ResNet-18 baseline.
- `src/hw5_swin.py`: Problem 2 scaffold for pretrained and scratch Swin experiments.
- `tools/make_hw5_plots.py`: helper script for generating loss-curve figures from exported CSVs.
- `Homework5_Problem1_ViT_vs_ResNet18.ipynb`: placeholder notebook for Problem 1 runs and analysis.
- `Homework5_Problem2_Swin_Finetuning.ipynb`: placeholder notebook for Problem 2 runs and analysis.
- `Results_Problem_1/`: Problem 1 CSV templates and output artifacts.
- `Results_Problem_2/`: Problem 2 CSV templates and output artifacts.
- `report/homework5_report.md`: report template covering every required section.
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
- The code and notebooks still need your final implementation, experiment runs, figures, and PDF report.
