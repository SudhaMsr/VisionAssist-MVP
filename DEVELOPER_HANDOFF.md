# Vision Assist — Developer Handoff

## Current repository
- `notebooks/vision-assist-v1.ipynb`: original V1 notebook.
- `notebooks/Vision_Assist_V2_Voice_Demo.ipynb`: cleaned V2 voice-demo notebook.
- `models/rod_yolov8n_50ep_best.pt`: existing trained YOLOv8n checkpoint.
- `models/rod_baseline_smoke_best.pt`: baseline checkpoint.
- `results/rod_yolov8n_50ep/`: validation metrics, plots and training history.
- `configs/rod_detection.yaml`: model configuration.

## Important status
- Do not retrain the model just to recreate the demo.
- The Kaggle runtime checked during handoff did not contain the generated voice-demo MP4 or audio files.
- The V2 notebook documents the intended detection-to-speech workflow. Run its demo cells using the existing checkpoint to regenerate outputs if required.
- The current demo is based on sample images, not a verified live-camera safety system.
- Review the README and notebook paths before further development.

## Status (2026-10-13)
The V2 voice demo is now the `vision_assist` package plus a thin notebook runner.
See README for setup, verified metrics, test results and limitations.
`results/voice_demo/` is legacy output from an earlier ad-hoc cell and is not
reproducible from current code; regenerate with `python -m vision_assist.demo`.
