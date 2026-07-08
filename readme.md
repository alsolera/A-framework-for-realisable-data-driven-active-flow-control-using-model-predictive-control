# A framework for realisable data-driven active flow control using model predictive control applied to a simplified truck wake

A. Solera-Rico, C. Sanmiguel Vila, S. Discetti. A framework for realisable
data-driven active flow control using model predictive control applied to a simplified
truck wake. Engineering Applications of Artificial Intelligence (2026).

Preprint [arXiv: 2510.11600.](https://arxiv.org/abs/2510.11600)

This repository contains the code and data accompanying the paper on surface-pressure-sensor-based
model predictive control (MPC) of the wake behind a 2D bluff body ("truck") using pulsed-jet
actuation. It covers the full pipeline: dataset ingestion, latent dynamics model training,
interpretable sensor selection (reducing 90 surface sensors down to a handful), closed-loop MPC on
the CFD simulation, and the robustness/generalisation analyses reported in the paper's Results
section and appendices.

## Repository structure

| Path | Contents                                                                                                                                                                                                                                                    |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `01_Data/` | HDF5 training/evaluation datasets (tracked in this repo).                                                                                                                                                                                                   |
| `02_Logs/`, `03_Checkpoints/`, `04_Results/` | Generated locally by the scripts below (MLflow logs, model checkpoints, figures/results). Not tracked in git, **except** for the one trained model checkpoint and SHAP-selected slim encoder shipped for reproducibility — see "What's included" below.     |
| `libs/` | Core library code: `models.py` (encoder/dynamics/decoder architectures), `data.py` (dataset loading), `MPC.py` (MPC controller), `environment.py` / `env_wrappers.py` (CFD environment via `gymprecice`), `test_encoder_predictor.py` (offline evaluation). |
| `Plots/` | Figure-generation scripts (latent space visualisation, pressure/wake distributions, control signal, prediction-horizon error, MPC video frames).                                                                                                            |
| `utils/` | Standalone inspection/preprocessing helpers (`inspect_h5.py`, `inspect_sensor_waveforms.py`, `plotCd_OF.py`, `preprocessor_no_fields.py`).                                                                                                                  |
| `physics-simulation-engine/` | OpenFOAM + preCICE case definition for the 2D truck CFD simulation (`fluid-openfoam/`), used by `gymprecice` during MPC runs.                                                                                                                               |
| `mlruns/` | MLflow tracking store (generated locally, not tracked).                                                                                                                                                                                                     |
| `parameters.py` | Central `Args` dataclass: training hyperparameters, data paths, derived run directories.                                                                                                                                                                    |
| `sensor_symmetry_map.json` | Sensor-index mapping used for symmetry-based data augmentation.                                                                                                                                                                                             |

## Setup

1.  **Clone the repository.**
2.  **Install dependencies** (see below).
3.  **Data:** the datasets referenced by `parameters.py` are already provided in `01_Data/`. To
    generate additional datasets from your own OpenFOAM runs, see `utils/preprocessor_no_fields.py`.
4.  **CFD environment (for closed-loop MPC runs on `JetTruck2DEnv`):**
    *   Install preCICE and OpenFOAM.
    *   The case in `physics-simulation-engine/fluid-openfoam/` is set up for coupling via
        `preciceDict`/`gymprecice-config.json`. Run `prerun.sh` once to decompose the mesh
        (`decomposePar`) before launching a coupled simulation.
    *   `libs/environment.py` assumes this case structure for reading forces and pressures.
5.  **MLflow:** experiment tracking is local-only. View logs with
    `mlflow ui --backend-store-uri ./02_Logs` from the project root.

## Dependencies

*   Python 3.8+
*   PyTorch, NumPy, Matplotlib, h5py, scikit-learn, SciPy, seaborn, PyWavelets
*   MLflow / mlflow-skinny
*   SHAP (for `rank_sensors_shap_and_train_slim.py`, `shap_robustness*.py`)
*   GitPython (used by `parameters.py` for run traceability)
*   gymprecice, pyprecice, gymnasium, Ofpp (for CFD-coupled runs)
*   tqdm, joblib

See `requirements.txt` for pinned versions; install with `pip install -r requirements.txt`.

## Core workflow

### 1. Model training

*   **Script:** `training.py`
*   **Description:** Trains the core model components: `TemporalEncoder`, `LatentDynamicsModel`, and `ForceDecoder`. It learns to encode sensor history into a latent space, predict latent state evolution, and decode forces from this latent space.
*   **Inputs:**
    *   HDF5 dataset(s) specified in `parameters.py` (`Args.datafile`, `Args.eval_dataset`).
    *   Hyperparameters defined in `parameters.py`.
*   **Outputs:**
    *   Model checkpoints (`*_encoder.pth.tar`, `*_dynamics.pth.tar`, `*_force.pth.tar`) in `03_Checkpoints/<case_name>/`.
    *   Latent space HDF5 file (containing latents, scaling info, and args) in `04_Results/<case_name>/`.
    *   Offline evaluation plots (latent error, force error, etc.) via `test_encoder_predictor.py`.
    *   MLflow logs in `02_Logs/<case_name>/`.
*   **Instructions:**
    1.  Ensure your HDF5 dataset is in `01_Data/`.
    2.  Configure `parameters.py` (especially `datafile`, `eval_dataset`, and model hyperparameters).
    3.  Run: `python training.py --comment "Your training run description"`
*   **Variant:** `training_noise_augmented.py` trains the same pipeline with additive Gaussian sensor
    noise injected during training, producing one model per noise level (used for the noise-robustness
    study — see §5 below).

### 2. Sensor selection / ranking & slim encoder training

*   **Script:** `rank_sensors_shap_and_train_slim.py`
*   **Description:**
    1.  **Phase 1 (SHAP Ranking):** Loads a pre-trained model. Uses SHAP (`GradientExplainer`) to explain the original encoder's output with respect to its input sensors. Ranks sensors based on their mean absolute SHAP values.
    2.  **Phase 2 (Slim Encoder):** Based on the top N sensors identified by SHAP, it trains a *new* `TemporalEncoder` (from scratch). This "slim" encoder only receives the SHAP-selected sensors as input and aims to match the original encoder's latent outputs.
*   **Inputs:**
    *   `base_model_strdate_identifier`: Timestamped ID of the model trained in step 1.
    *   Configuration dataclasses at the top of the script (`GLOBAL_CONFIG_SHAP`, `SHAP_CONFIG`, `SLIM_CONFIG_SHAP`).
*   **Outputs:**
    *   Checkpoint for the SHAP-based slim encoder in `03_Checkpoints/<original_case_name>/shap_optim_runs/<run_name>/`.
    *   SHAP importance plots, selected sensor indices, SHAP value map in `04_Results/<original_case_name>/shap_optim_runs/<run_name>/`.
    *   MLflow logs in `02_Logs/<original_case_name>/shap_optim_runs/`.
*   **Instructions:**
    1.  Edit `rank_sensors_shap_and_train_slim.py`:
        *   Set `GLOBAL_CONFIG_SHAP.base_model_strdate_identifier`.
        *   Adjust `SHAP_CONFIG` (number of samples, top N sensors) and `SLIM_CONFIG_SHAP`.
    2.  Run: `python rank_sensors_shap_and_train_slim.py`

### 3. Model predictive control (MPC) on CFD simulation

*   **Script:** `MPC_onFOM.py`
*   **Description:** Runs the MPC controller defined in `libs/MPC.py` on the `JetTruck2DEnv` (CFD simulation). The MPC uses a trained encoder (either original/full or a slim one), dynamics model, and force decoder to optimize control actions.
*   **Inputs:**
    *   `original_model_id`: Timestamped ID of the model whose dynamics and force decoder will be used.
    *   Flags and paths to specify whether to use a slim encoder and its artifacts (from step 2).
    *   MPC horizon, optimization parameters, cost function weights.
    *   Optional sensor-noise (`SENSOR_NOISE_SIGMA`) and actuator-delay (`ACTUATOR_DELAY`) settings, used for the robustness study in §5.
*   **Outputs:**
    *   Detailed HDF5 log of the MPC run (actions, rewards, predicted/actual forces, etc.) and summary plots, written to `gymprecice-run/<run_name>/` (relative to the working directory; `<run_name>` encodes the model ID, controller type, and timestamp).
    *   CFD simulation results in the `gymprecice-run` subfolder created by `JetTruck2DEnv`.
*   **Instructions:**
    1.  Edit `MPC_onFOM.py`:
        *   Set `original_model_id`.
        *   Set `USE_SLIM_ENCODER` flag.
        *   If `USE_SLIM_ENCODER=True`, accurately set `_case_name_from_original_id`, `_sensor_optim_run_name`, and thus the paths `SLIM_ENCODER_CKPT_PATH` and `SELECTED_INDICES_PATH` to point to the artifacts from step 2.
        *   Configure MPC parameters.
    2.  Ensure your preCICE setup and OpenFOAM case are ready for `JetTruck2DEnv` (see Setup §4).
    3.  Run: `python MPC_onFOM.py`

## Reproducing paper results & robustness analyses

These scripts post-process the outputs of steps 1–3 above (trained checkpoints, MPC run HDF5 files)
to produce the Results-section figures and appendix analyses. They are analysis/plotting scripts,
not part of the core training/control pipeline, and most expect the path to a specific prior run —
edit the constants at the top of each file (usually marked `# USER:` or `RESULT_FILES =`) to point at
your own runs.

**Sensor-subset evaluation**
*   `evaluate_sensor_subsets.py` — trains and evaluates slim encoders for a range of sensor-subset sizes, producing the open-loop prediction-error-vs-number-of-sensors curve.

**Out-of-distribution generalisation** (*"Model performance outside the training frequency band"* appendix)
*   `evaluate_ood_chirp.py` — evaluates the full and slim encoders on an extended-frequency chirp dataset that exceeds the training frequency band.

**Sensor-noise & actuator-delay robustness** (*"Closed-loop robustness to sensor noise and actuator delay"* appendix)
*   `training_noise_augmented.py` — trains the full pipeline with additive Gaussian sensor noise injected during training, for a sweep of noise levels.
*   `evaluate_noise_robustness.py` — evaluates the (clean-trained) 4-sensor slim encoder under test-time-only Gaussian sensor noise.
*   `evaluate_noise_robustness_augmented.py` — evaluates noise-augmented slim encoders (from `training_noise_augmented.py`) and compares against the clean-model degradation curve.
*   `compare_mpc_noise.py` — compares closed-loop MPC performance (Cd, Cl, control action) across MPC runs at different sensor-noise levels.
*   `compare_mpc_delay.py` — same comparison, across different actuator-delay levels.

**SHAP-based sensor-selection robustness** (*"Robustness of the SHAP-based sensor selection"* appendix)
*   `shap_robustness.py` — repeats SHAP `GradientExplainer` analysis over multiple random seeds to assess ranking stability (top-k stability scores, Spearman rank correlation across runs).
*   `shap_robustness_composite.py` — builds the composite 3-panel figure: SHAP importance stability, local correlation structure around the selected sensors, and leave-one-out retraining error.

**MPC optimizer sensitivity / run comparison**
*   `mpc_lr_sensitivity.py` — offline sensitivity analysis of the MPC's internal optimizer learning rate on converged cost/optimal action.
*   `compare_mpc_runs.py` — compares two or more MPC-on-FOM runs (e.g. different optimizer learning rates) to show equivalent closed-loop drag reduction.

## Plotting & inspection utilities

*   **`Plots/`** — figure-generation scripts used to produce paper figures: latent-space visualisation (`visualize_latent*.py`), pressure/wake-velocity distribution comparisons (`plot_p_distributions.py`, `plot_wake_probe_comparison.py`, `plot_back_pressure_comparison.py`), control-signal spectrograms (`plot_control_signal.py`), prediction-horizon error tables (`plot_horizon_error.py`), prediction distributions (`plot_prediction_distribution.py`), and MPC run video frames (`MPC_video_frames.py`).
*   **`utils/`** — standalone helpers: `inspect_h5.py` (generic HDF5 inspection), `inspect_sensor_waveforms.py` (diagnostic plot of raw sensor waveforms across noise levels), `plotCd_OF.py` (plot drag/lift coefficients directly from OpenFOAM `forceCoeffs` output), `preprocessor_no_fields.py` (build a training-ready HDF5 dataset from raw OpenFOAM post-processing output).

Several of these scripts hardcode example paths under `../gymprecice-run/...` pointing at a specific
past run — these are placeholders (marked with a `USER: PLEASE UPDATE THESE PATHS` comment) and must
be edited to point at your own `MPC_onFOM.py` / CFD run output before use.

**What's included vs. regenerated:** to keep the repository lean, this release ships only the two
trained-model artifacts that are expensive/stochastic to reproduce exactly — the full 90-sensor model
checkpoint (`03_Checkpoints/jet_2Dtruck_20250307_FMsignal_50000/20250906_19_38_58_*.pth.tar`) and the
SHAP-selected 4-sensor slim encoder + its selected-sensor-indices file
(`03_Checkpoints/.../shap_optim_runs/<run_name>/..._phase2_slim_shap_encoder_slim_shap.pth.tar` and
`04_Results/.../shap_optim_runs/<run_name>/..._shap_selected_indices.txt`). Everything else —
MPC run HDF5 logs (`gymprecice-run/...`, generated by `MPC_onFOM.py`), the sensor-subset evaluation
sweep, SHAP stability cache, OOD-chirp results, noise-augmented checkpoints, the full-model
latent-space HDF5s, and the raw OpenFOAM `postProcessing/` probe output consumed by
`plot_back_pressure_comparison.py` / `plot_wake_probe_comparison.py` / `plot_p_distributions.py`
(generated as part of any `MPC_onFOM.py` CFD run) — is deterministic (given the same seed and
checkpoint) to regenerate by running the corresponding script in §"Reproducing paper results &
robustness analyses" against the shipped
checkpoint, and is not stored in the repository.

## Notes

*   The `*_CONFIG` dataclasses at the top of the main scripts (`rank_sensors_shap_and_train_slim.py`, `MPC_onFOM.py`) are the primary way to configure runs. **Remember to set `base_model_strdate_identifier` or similar path configurations in these scripts before running.**
*   Path constructions, especially for slim encoder artifacts, depend on consistent naming conventions from previous steps. Double-check these paths if you encounter `FileNotFoundError`.
*   The CFD interaction via `JetTruck2DEnv` requires a functional preCICE and OpenFOAM setup. Ensure this is working independently before running CFD-coupled scripts.

## License

This repository is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license — see [`LICENSE`](LICENSE). You are free to share and adapt the code and data, including for commercial purposes, provided you give appropriate credit.
