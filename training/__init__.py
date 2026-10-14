"""Fine-tuning pipeline for VisionAssist (experiment v1).

Separate from the app (vision_assist/): nothing here is imported by the app, and
the shipped model models/rod_yolov8n_50ep_best.pt is only ever read, never written.
See notebooks/finetune_v1.ipynb for the Kaggle run order.
"""
