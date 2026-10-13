# Archive

Superseded files kept for provenance only. Nothing in `vision_assist/`, `tests/`,
`scripts/` or the V2 notebook uses them.

| Path | What it is | Why archived |
|---|---|---|
| `notebooks/vision-assist-v1-scratch.ipynb` | Formerly `notebooks/vision-assist-v1 (3).ipynb`. An exploratory copy of the V1 notebook. | Its 26 "Phase 2" cells were never executed. The executed cells are git-upload chores plus cell 49, which produced `voice_demo_legacy/`. |
| `results/voice_demo_legacy/` | Formerly `results/voice_demo/`. Demo output from that ad-hoc cell. | It predates the `vision_assist` rule engine (it used alphabetical class order with no repeat suppression), and the current code cannot reproduce it. |

The executed training run, which is the source of the validation metrics, stays in
`notebooks/vision-assist-v1.ipynb`.
