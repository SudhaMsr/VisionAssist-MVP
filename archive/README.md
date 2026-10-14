# Archive

Superseded files kept for provenance only. Nothing in `vision_assist/`, `tests/`,
`scripts/` or the V2 notebook uses them.

| Path | What it is | Why archived |
|---|---|---|
| `notebooks/vision-assist-v1-scratch.ipynb` | Formerly `notebooks/vision-assist-v1 (3).ipynb`. An exploratory copy of the V1 notebook. | Its 26 "Phase 2" cells were never executed. The executed cells are git-upload chores plus cell 49, which produced `voice_demo_legacy/`. |
| `results/voice_demo_legacy/` | Formerly `results/voice_demo/`. Demo output from that ad-hoc cell. | It predates the `vision_assist` rule engine (it used alphabetical class order with no repeat suppression), and the current code cannot reproduce it. |
| `results/voice_demo_v2_recovered/` | A recovered export, restored on 2026-10-14, of the same 8 images (IMG_20816–20823) and the same "Caution. Car detected." wording: 8 alert clips, the voice WAV, a voice MP4, a silent MP4, `demo_results.json` and `spoken_alerts.txt`. Its 8 annotated frames were byte-identical to `voice_demo_legacy/frame_0N.jpg`, so they were not copied again. | Same reasons as `voice_demo_legacy/`. The MP4 is also defective: the WAV is 24.9 s but the MP4 stops at 16.0 s (8 frames × 2 s), and speech is spaced about 3.1 s apart, so it drifts out of step with the frames. The audio cuts off during alert 6, and alerts 7 (Manhole) and 8 are never heard. Use `vision_assist_v2_voice.wav` for the complete audio. |

The executed training run, which is the source of the validation metrics, stays in
`notebooks/vision-assist-v1.ipynb`.
