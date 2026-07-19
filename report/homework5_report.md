# Homework 5: Vision Transformers for Image Classification

**Name:** Gilberto Feliu  
**Student ID:** 801257813  
**Assignment:** Homework 5  
**GitHub Repository:** `ADD_PUBLIC_GITHUB_LINK_HERE`

## Late Submission Note

This homework was submitted late because I had a family medical emergency. My grandmother has pancreatic cancer and was hospitalized during the week of the deadline, which disrupted the time I had planned for completing and packaging the assignment.

## Dataset and Experimental Setup

Both problems use the CIFAR-100 dataset from `torchvision.datasets.CIFAR100`. State the train/test preprocessing, normalization, augmentation choices, hardware used, CPU fallback support, and the exact package versions used for the runs reported below.

## Problem 1: Vision Transformer from Scratch vs. ResNet-18

### Objective

Summarize the assignment goal in your own words: compare several ViT configurations against a ResNet-18 baseline on CIFAR-100 after 10 epochs, and analyze accuracy, parameter count, FLOPs, and runtime.

### Configurations

Document at least four ViT configurations and one ResNet-18 baseline.

| Model | Patch Size | Embedding Dim | Depth | Heads | MLP Dim | Epochs | Batch Size | LR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT-A | 4 | 256 | 4 | 4 | 1024 | 10 | 64 | 0.001 |
| ViT-B | 4 | 512 | 8 | 8 | 2048 | 10 | 64 | 0.001 |
| ViT-C | 8 | 256 | 4 | 4 | 1024 | 10 | 64 | 0.001 |
| ViT-D | 8 | 512 | 8 | 8 | 2048 | 10 | 64 | 0.001 |
| ResNet-18 | - | - | - | - | - | 10 | 64 | 0.001 |

### Implementation Summary

Describe:

- patch embedding and class token handling
- positional embedding strategy
- transformer encoder block design
- classifier head
- ResNet-18 baseline setup
- optimizer, loss, scheduler, and evaluation logic

### Results Table

Copy in the final values from `Results_Problem_1/problem1_summary.csv`.

| Model | Params | FLOPs / Forward | Train Time / Epoch (s) | Final Val Loss | Test Accuracy (%) |
|---|---:|---:|---:|---:|---:|
| ViT-A | TODO | TODO | TODO | TODO | TODO |
| ViT-B | TODO | TODO | TODO | TODO | TODO |
| ViT-C | TODO | TODO | TODO | TODO | TODO |
| ViT-D | TODO | TODO | TODO | TODO | TODO |
| ResNet-18 | TODO | TODO | TODO | TODO | TODO |

### Analysis

Discuss:

- which ViT configuration performed best
- which model had the best accuracy/compute tradeoff
- how patch size changed token count and cost
- why some ViT setups underperformed ResNet-18 after only 10 epochs
- how parameter count and FLOPs related to runtime and accuracy

Reference your exported figures:

- `Results_Problem_1/problem1_loss_curves.png`
- any additional comparison figure you generate

## Problem 2: Fine-Tuning Pretrained Swin Transformers vs. Training from Scratch

### Objective

Summarize the assignment goal in your own words: compare pretrained Swin-Tiny and Swin-Small head-only fine-tuning against a scratch Swin-style model on CIFAR-100 after 5 epochs.

### Implementation Summary

Describe:

- how you loaded pretrained models from Hugging Face
- how you replaced the classification head for 100 classes
- how you froze the backbone
- training settings for pretrained models
- scratch model design and training settings

### Results Table

Copy in the final values from `Results_Problem_2/problem2_summary.csv`.

| Model | Pretrained | Frozen Backbone | Params | Train Time / Epoch (s) | Final Val Loss | Test Accuracy (%) |
|---|---|---|---:|---:|---:|---:|
| Swin-Tiny | Yes | Yes | TODO | TODO | TODO | TODO |
| Swin-Small | Yes | Yes | TODO | TODO | TODO | TODO |
| Scratch Swin | No | No | TODO | TODO | TODO | TODO |

### Analysis

Discuss:

- whether fine-tuning beat scratch training after 5 epochs
- whether Swin-Small justified its higher complexity
- benefits and drawbacks of frozen-backbone transfer learning
- reasons a scratch model might lag on CIFAR-100 in such a short schedule

Reference your exported figures:

- `Results_Problem_2/problem2_training_curves.png`
- any additional summary figure you generate

## Conclusion

State the main takeaways from both problems:

- ViT vs. ResNet-18 tradeoffs
- fine-tuning vs. scratch tradeoffs
- what you would change with more time or compute

