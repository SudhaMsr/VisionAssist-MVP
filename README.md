# Vision Assist MVP

Experimental assistive-vision prototype using YOLO object detection.

## Current baseline
- Dataset: ROD obstacle detection dataset
- Model: YOLOv8n pretrained baseline
- Preliminary one-epoch validation mAP50: 0.6746
- Preliminary one-epoch validation mAP50-95: 0.4664

These are preliminary validation results, not final test-set results.
This prototype is experimental and is not safety-grade navigation software.

## Contents
- `notebooks/`: training notebook
- `configs/`: dataset configuration
- `models/`: saved model checkpoints

## Dataset
The dataset is obtained separately from Kaggle. Dataset images and labels
are not included in this repository.

## Reproducibility
Review the notebook for preprocessing, training and validation steps.
Check dataset paths and dependencies before rerunning.
