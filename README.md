# AIRL4MMV

Standalone publication snapshot for adversarial inverse reinforcement learning on mixed-mode ventilation control.

## Contents

- `src/`: core model, environment, and AIRL training code
- `configs/`: reproducible experiment settings
- `notebooks/`: publication-facing training and evaluation notebooks
- `models/trained_model_new_full.pth`: required building dynamics checkpoint
- `l14_merged_data_with_rain.csv`: dataset used by the published notebooks

## Supported workflow

The published notebook flow is intentionally narrow. The maintained reward variants are:

- `temps_hold_dwell_prev_time_gru`
- `temps_hold_dwell_prev_time_comfort_zonal_gru`
- `temps_hold_dwell_prev_time_energy_zonal_gru`
- `temps_hold_dwell_prev_time_energy_comfort_zonal_gru`

Training notebooks generate policy and reward checkpoints under `notebooks/outputs/`. The comparison notebooks expect those checkpoints to exist inside this repository, not in the parent project.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_standalone_repo.py
```

Open the notebooks from this folder root. They already resolve project paths locally and no longer require the parent `AIRL4MMV_debug` tree.

## Notes

- This snapshot keeps the AIRL environment and current notebook workflow, but removes generated outputs and cache files.
- The public repo includes only the dynamics checkpoint required by the notebooks. Training and evaluation outputs should be regenerated inside `notebooks/outputs/`.
