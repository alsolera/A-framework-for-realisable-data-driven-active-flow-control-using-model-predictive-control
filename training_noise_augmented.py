# training_noise_augmented.py
#
# Trains the full pipeline (TemporalEncoder + LatentDynamicsModel + ForceDecoder)
# with additive Gaussian sensor noise injected during training, for a set of
# noise levels alpha. For each alpha, a complete model is trained from scratch.
#
# Noise characterisation:
#       sigma_noise = alpha   (data is already z-scored, so alpha is directly
#                              the noise-to-signal ratio w.r.t. sensor std)
#
# Noise is injected on the FULL sensor time series BEFORE sequence creation,
# so every sequence that contains timestep t sees the same noise realisation
# for t. A new noise realisation is drawn each epoch (data augmentation).
# Validation always uses clean inputs for a fair early-stopping criterion.
#
# After training, slim encoders are retrained on top of each noise-trained
# full model (identical procedure to evaluate_sensor_subsets.py), and then
# evaluated at the matching noise level.
#
# Outputs:
#   03_Checkpoints/<case>/<modelname_noise{pct}pct>/   -- full model checkpoints
#   04_Results/.../noise_augmented_full/               -- slim ckpts, results json,
#                                                         combined plot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import mlflow
import time
import os
import h5py
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from libs.models import TemporalEncoder, LatentDynamicsModel, ForceDecoder, compute_total_test_loss
from libs.data import get_prepared_data, prepare_raw_data, build_dataloader_from_scaled, loadData
from libs.test_encoder_predictor import find_model_paths, load_checkpoint
from parameters import Args

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Noise levels for FULL MODEL training (alpha=0 is the existing clean model)
NOISE_ALPHAS_TRAIN = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

# Base clean model used only to recover SHAP rankings and global sensor range
BASE_MODEL_ID   = "20250906_19_38_58"
CHECKPOINTS_DIR = "03_Checkpoints"
RESULTS_DIR     = "04_Results"
NUM_SENSORS     = 4   # slim encoder sensors

# Slim encoder training (mirrors evaluate_sensor_subsets.py)
SLIM_EPOCHS = 100
SLIM_LR     = 1e-3
SLIM_BATCH  = 256

SEED = 4321
SEEDeval = 31416
CUDA = True
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_raw_forces(args_for_data, sel_coefs=('Cd', 'Cl')):
    class _Args:
        datafile               = args_for_data.datafile
        augment_with_symmetry  = False
    (_, forces_unscaled, *_) = loadData(_Args, sel_coefs=list(sel_coefs))
    return forces_unscaled


def train_full_model_with_noise(args, device, sensor_noise_sigma, strdate_tag):
    """
    Train the full pipeline (encoder + dynamics + force_decoder) from scratch
    with Gaussian noise injected on the sensor time series before sequence
    creation, so all sequences sharing timestep t see the same noise at t.
    A fresh noise realisation is drawn each epoch.
    Validation always uses clean inputs.

    Returns: (encoder, dynamics_model, force_decoder, ckp_dir, model_name)
    """
    # --- Load and scale raw training data ONCE (expensive I/O step) ---
    p_scaled_train, control_scaled_train, forces_scaled_train, file_end_indices_train = (
        prepare_raw_data(args))

    # Infer dims from a single clean dataloader build
    dl_tmp, _ = build_dataloader_from_scaled(
        p_scaled_train, control_scaled_train, forces_scaled_train,
        file_end_indices_train, args, device,
        noise_sigma=0.0, shuffle_train=False)
    in_dim     = dl_tmp.dataset[0][0].shape[-1]
    action_dim = dl_tmp.dataset[0][1].shape[-1]
    del dl_tmp

    # --- Clean evaluation dataloader (built once, never noised) ---
    eval_args                       = deepcopy(args)
    eval_args.datafile              = args.eval_dataset
    eval_args.n_test                = 0
    eval_args.augment_with_symmetry = False
    dataloader_eval, _              = get_prepared_data(eval_args, device,
                                                        shuffle_train=False,
                                                        shuffle_test=False)

    # --- Models ---
    encoder = TemporalEncoder(
        input_dim=in_dim, latent_dim=args.latent_dim,
        hidden_dim=args.enc_hidden_dim, num_layers=args.n_layers
    ).to(device)
    dynamics_model = LatentDynamicsModel(
        latent_dim=args.latent_dim, action_dim=action_dim,
        hidden_dim=args.dyn_hidden_dim, use_residual=args.residual_predictor
    ).to(device)
    force_decoder = ForceDecoder(
        latent_dim=args.latent_dim, hidden_dim=args.forces_hidden_dim,
        dropout_rate=args.forces_dropout, arch=args.force_decoder_arch
    ).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(dynamics_model.parameters()) +
        list(force_decoder.parameters()),
        lr=args.lr
    )

    model_name = (f'{strdate_tag}_LSTM_dim{args.latent_dim}_lb{args.lookback}'
                  f'_l{args.n_layers}_h{args.dyn_hidden_dim}_dr{args.dropout}'
                  f'_lr{args.lr}_bs{args.batch_size}'
                  f'_noise{args.sensor_noise_alpha_pct:.2f}pct')
    args.modelname = model_name

    ckp_dir = Path(CHECKPOINTS_DIR) / args.case / model_name
    ckp_dir.mkdir(parents=True, exist_ok=True)

    bestloss  = float('inf')
    rng       = np.random.default_rng(SEED)
    alpha_pct = args.sensor_noise_alpha_pct

    print(f"\n{'='*65}")
    print(f"Training full model | sigma_noise={sensor_noise_sigma:.5f} | {model_name}")
    print(f"{'='*65}")

    # --- MLflow setup ---
    mlflow.set_tracking_uri(f"file://{str(Path(args.logdir).resolve())}")
    mlflow.set_experiment(f"{args.case} LSTM noise_augmented")
    print(f'MLflow: mlflow ui -p 5001 --backend-store-uri file://{os.getcwd()}/02_Logs/{args.case}')

    with mlflow.start_run(run_name=model_name):
        public_args = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        mlflow.log_params(public_args)
        mlflow.set_tag("noise_alpha_pct", f"{alpha_pct:.2f}%")
        mlflow.set_tag("noise_sigma",     f"{sensor_noise_sigma:.5f}")
        mlflow.set_tag("run_comment",     f"Full pipeline noise augmentation alpha={alpha_pct:.2f}%")

        for epoch in range(1, args.epochs + 1):

            # --- Rebuild training dataloader with fresh noise each epoch ---
            dataloader_train, _ = build_dataloader_from_scaled(
                p_scaled_train, control_scaled_train, forces_scaled_train,
                file_end_indices_train, args, device,
                noise_sigma=sensor_noise_sigma, rng=rng,
                shuffle_train=True)

            encoder.train(); dynamics_model.train(); force_decoder.train()

            epoch_total_loss    = 0.0
            epoch_mse_loss      = 0.0
            epoch_var_loss      = 0.0
            epoch_cov_loss      = 0.0
            epoch_cd_cl_loss    = 0.0
            dynamics_loss_count = 0
            force_loss_count    = 0
            num_batches_train   = len(dataloader_train)

            for batch in dataloader_train:
                s_t, a_seq, s_t1, C_seq = batch
                s_t   = s_t.to(device)
                a_seq = a_seq.to(device)
                s_t1  = s_t1.to(device)
                C_seq = C_seq.to(device)

                # --- Control noise (keep existing behaviour) ---
                if args.control_noise_std > 0.0:
                    a_seq = a_seq + torch.randn_like(a_seq) * args.control_noise_std

                if args.recursive_train_steps == 1:
                    a_seq = a_seq[:, 0, :]
                    C_seq = C_seq[:, 0, :]

                # --- Forward pass ---
                z_t     = encoder(s_t)
                z_preds = [z_t]
                z       = z_t
                if args.recursive_train_steps == 1:
                    z = dynamics_model(z, a_seq)
                    z_preds.append(z)
                else:
                    for step in range(args.recursive_train_steps):
                        z = dynamics_model(z, a_seq[:, step, :])
                        z_preds.append(z)

                z_targets    = [None] * len(z_preds)
                z_targets[0] = z_t
                for step in range(args.recursive_train_steps):
                    target_input = s_t1[:, step, :]
                    if target_input.ndim == 2:
                        target_input = target_input.unsqueeze(1)
                    z_targets[step + 1] = encoder(target_input)

                # --- Loss ---
                batch_total_dynamics_loss = torch.tensor(0.0, device=device)
                batch_total_force_loss    = torch.tensor(0.0, device=device)
                batch_mse_sum = batch_var_sum = batch_cov_sum = batch_cdcl_sum = 0.0
                num_dynamics_steps = 0
                num_force_steps    = 0

                for i, z_pred in enumerate(z_preds[1:], start=1):
                    is_first = (i == 1)
                    is_last  = (i == args.recursive_train_steps)
                    target   = z_targets[i]

                    apply_dyn = (args.loss_mode == "all" or
                                 (args.loss_mode == "first_and_last" and (is_first or is_last)) or
                                 (args.loss_mode == "last" and is_last))

                    if apply_dyn:
                        batch_size, latent_dim = z_pred.shape
                        mse_i = torch.mean((z_pred - target) ** 2)
                        z_c   = z_pred - z_pred.mean(dim=0, keepdim=True)
                        var_i = torch.mean(torch.relu(1.0 - torch.sqrt(z_c.var(dim=0, unbiased=True) + 1e-8)))
                        cov_i = torch.tensor(0.0, device=device)
                        if batch_size > 1 and latent_dim > 1:
                            cov_m = (z_c.T @ z_c) / (batch_size - 1)
                            mask  = ~torch.eye(latent_dim, dtype=torch.bool, device=device)
                            cov_i = (cov_m[mask] ** 2).mean()
                        batch_total_dynamics_loss += (args.lambda_mse * mse_i +
                                                      args.lambda_var * var_i +
                                                      args.lambda_cov * cov_i)
                        batch_mse_sum += mse_i.item()
                        batch_var_sum += var_i.item()
                        batch_cov_sum += cov_i.item()
                        num_dynamics_steps += 1

                    c_true_i = C_seq if args.recursive_train_steps == 1 else C_seq[:, i - 1, :]
                    loss_fn  = nn.SmoothL1Loss()
                    c_pred_i = force_decoder(z_pred)
                    cd_cl_i  = loss_fn(c_pred_i, c_true_i)
                    if args.force_decoder_noise_std > 0.0:
                        z_noisy = z_pred.detach() + torch.randn_like(z_pred) * args.force_decoder_noise_std
                        cd_cl_i = cd_cl_i + loss_fn(force_decoder(z_noisy), c_true_i)
                    batch_total_force_loss += args.lambda_cd_cl * cd_cl_i
                    batch_cdcl_sum         += cd_cl_i.item()
                    num_force_steps        += 1

                avg_dyn   = batch_total_dynamics_loss / num_dynamics_steps if num_dynamics_steps > 0 else torch.tensor(0.0, device=device)
                avg_force = batch_total_force_loss    / num_force_steps    if num_force_steps    > 0 else torch.tensor(0.0, device=device)
                total     = avg_dyn + avg_force

                optimizer.zero_grad()
                total.backward()
                optimizer.step()

                epoch_total_loss += total.item()
                if num_dynamics_steps > 0:
                    epoch_mse_loss   += batch_mse_sum  / num_dynamics_steps
                    epoch_var_loss   += batch_var_sum  / num_dynamics_steps
                    epoch_cov_loss   += batch_cov_sum  / num_dynamics_steps
                    dynamics_loss_count += 1
                if num_force_steps > 0:
                    epoch_cd_cl_loss += batch_cdcl_sum / num_force_steps
                    force_loss_count += 1

            # --- Epoch averages ---
            avg_epoch_total_loss  = epoch_total_loss  / num_batches_train
            avg_epoch_mse_loss    = epoch_mse_loss    / dynamics_loss_count if dynamics_loss_count > 0 else 0.0
            avg_epoch_var_loss    = epoch_var_loss    / dynamics_loss_count if dynamics_loss_count > 0 else 0.0
            avg_epoch_cov_loss    = epoch_cov_loss    / dynamics_loss_count if dynamics_loss_count > 0 else 0.0
            avg_epoch_cd_cl_loss  = epoch_cd_cl_loss  / force_loss_count    if force_loss_count    > 0 else 0.0

            mlflow.log_metric("train_loss",        avg_epoch_total_loss,  step=epoch)
            mlflow.log_metric("train_mse_loss",    avg_epoch_mse_loss,    step=epoch)
            mlflow.log_metric("train_var_loss",    avg_epoch_var_loss,    step=epoch)
            mlflow.log_metric("train_cov_loss",    avg_epoch_cov_loss,    step=epoch)
            mlflow.log_metric("train_cd_cl_loss",  avg_epoch_cd_cl_loss,  step=epoch)

            # --- Validation (CLEAN inputs no noise) ---
            encoder.eval(); dynamics_model.eval(); force_decoder.eval()
            test_loss = test_mse = test_var = test_cov = test_cd_cl = 0.0
            num_batches_test = len(dataloader_eval)

            with torch.no_grad():
                for s_t_v, a_v, s_t1_v, C_v in dataloader_eval:
                    s_t_v, a_v, s_t1_v, C_v = (s_t_v.to(device), a_v.to(device),
                                                 s_t1_v.to(device), C_v.to(device))
                    if args.recursive_train_steps == 1:
                        a_v = a_v[:, 0, :]
                        C_t = C_v[:, 0, :]
                    else:
                        C_t = C_v[:, -1, :]

                    z_v    = encoder(s_t_v)
                    tgt_in = s_t1_v[:, -1, :].unsqueeze(1)
                    z_real = encoder(tgt_in)
                    z_pred = z_v
                    if args.recursive_train_steps == 1:
                        z_pred = dynamics_model(z_v, a_v)
                    else:
                        for step in range(args.recursive_train_steps):
                            z_pred = dynamics_model(z_pred, a_v[:, step, :])
                    c_pred = force_decoder(z_pred)
                    vl, val_mse, val_var, val_cov, val_cdcl = compute_total_test_loss(
                        z_pred, z_real, c_pred, C_t,
                        args.lambda_mse, args.lambda_var,
                        args.lambda_cov, args.lambda_cd_cl,
                        return_components=True)
                    test_loss    += vl.item()
                    test_mse     += val_mse
                    test_var     += val_var
                    test_cov     += val_cov
                    test_cd_cl   += val_cdcl

            loss_test         = test_loss    / num_batches_test
            loss_mse_test     = test_mse     / num_batches_test
            loss_var_test     = test_var     / num_batches_test
            loss_cov_test     = test_cov     / num_batches_test
            loss_cd_cl_test   = test_cd_cl   / num_batches_test

            mlflow.log_metric("test_loss",       loss_test,       step=epoch)
            mlflow.log_metric("test_mse_loss",   loss_mse_test,   step=epoch)
            mlflow.log_metric("test_var_loss",   loss_var_test,   step=epoch)
            mlflow.log_metric("test_cov_loss",   loss_cov_test,   step=epoch)
            mlflow.log_metric("test_cd_cl_loss", loss_cd_cl_test, step=epoch)

            # --- Save best model (last 10% of epochs, mirrors training.py) ---
            if loss_test < bestloss and epoch > (args.epochs * 0.9 - 1):
                bestloss = loss_test
                ckp_enc = str(ckp_dir / f"{model_name}_encoder.pth.tar")
                ckp_dyn = str(ckp_dir / f"{model_name}_dynamics.pth.tar")
                ckp_frc = str(ckp_dir / f"{model_name}_force.pth.tar")
                torch.save({'state_dict': encoder.state_dict()},       ckp_enc)
                torch.save({'state_dict': dynamics_model.state_dict()},ckp_dyn)
                torch.save({'state_dict': force_decoder.state_dict()}, ckp_frc)
                with open(ckp_dir / f"{model_name}_args.json", 'w') as f:
                    json.dump(asdict(args), f, indent=4)
                mlflow.log_artifact(ckp_enc, artifact_path="checkpoints")
                mlflow.log_artifact(ckp_dyn, artifact_path="checkpoints")

            print(f"  Epoch {epoch}/{args.epochs} | "
                  f"train={avg_epoch_total_loss:.5f} | "
                  f"val={loss_test:.5f} | best={bestloss:.5f}")

    # Load best weights before returning
    load_checkpoint(encoder,       str(ckp_dir / f"{model_name}_encoder.pth.tar"),   device)
    load_checkpoint(dynamics_model,str(ckp_dir / f"{model_name}_dynamics.pth.tar"),  device)
    load_checkpoint(force_decoder, str(ckp_dir / f"{model_name}_force.pth.tar"),     device)
    encoder.eval(); dynamics_model.eval(); force_decoder.eval()

    return encoder, dynamics_model, force_decoder, ckp_dir, model_name


def train_slim_encoder(slim_encoder, encoder_target,
                       p_scaled_train, control_scaled_train, forces_scaled_train,
                       file_end_indices_train, loader_args_train,
                       selected_indices, noise_sigma, device, ckpt_path, rng):
    """
    Train slim encoder via distillation.
    Each epoch a fresh noisy dataloader is built from the raw scaled arrays
    so all sequences sharing timestep t see the same noise realisation for t.
    Validation uses clean inputs.
    """
    optimizer = optim.Adam(slim_encoder.parameters(), lr=SLIM_LR)
    best_val  = float('inf')

    # Build clean validation loader once
    _, dl_val = build_dataloader_from_scaled(
        p_scaled_train, control_scaled_train, forces_scaled_train,
        file_end_indices_train, loader_args_train, device,
        noise_sigma=0.0, shuffle_train=False)

    pbar = tqdm(range(1, SLIM_EPOCHS + 1),
                desc=f"Slim encoder (sigma={noise_sigma:.5f})", leave=False)
    for epoch in pbar:
        # Fresh noisy dataloader each epoch
        dl_train_noisy, _ = build_dataloader_from_scaled(
            p_scaled_train, control_scaled_train, forces_scaled_train,
            file_end_indices_train, loader_args_train, device,
            noise_sigma=noise_sigma, rng=rng, shuffle_train=True)

        slim_encoder.train()
        for s_t_full, _, _, _ in dl_train_noisy:
            s_t_full = s_t_full.to(device)
            s_t_slim = s_t_full[:, :, selected_indices]
            optimizer.zero_grad()
            with torch.no_grad():
                z_target = encoder_target(s_t_full)
            loss = F.mse_loss(slim_encoder(s_t_slim), z_target)
            loss.backward(); optimizer.step()

        # Clean validation
        slim_encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s_t_full_v, _, _, _ in dl_val:
                s_t_slim_v = s_t_full_v[:, :, selected_indices].to(device)
                z_tgt_v    = encoder_target(s_t_full_v.to(device))
                val_loss  += F.mse_loss(slim_encoder(s_t_slim_v), z_tgt_v).item()
        avg_val = val_loss / len(dl_val)
        pbar.set_postfix({'best_val': f'{best_val:.6f}'})
        if avg_val < best_val:
            best_val = avg_val
            torch.save({'state_dict': slim_encoder.state_dict()}, ckpt_path)

    print(f"  Slim encoder done. Best val: {best_val:.6f} -> {ckpt_path}")


def evaluate_force_prediction(slim_encoder, force_decoder, dataloader, selected_indices,
                              raw_forces_gt, forces_mean, forces_std, device):
    """Clean forward pass noise already baked into the dataloader."""
    slim_encoder.eval(); force_decoder.eval()
    preds_list = []
    with torch.no_grad():
        for (s_t_full, _, _, _) in DataLoader(dataloader.dataset, batch_size=512, shuffle=False):
            s_t = s_t_full[:, :, selected_indices].to(device)
            preds_list.append(force_decoder(slim_encoder(s_t)).cpu())

    y_pred = torch.cat(preds_list).numpy() * forces_std + forces_mean
    offset = dataloader.dataset.tensors[0].shape[1] - 1
    y_true = raw_forces_gt[offset: offset + len(y_pred)]
    if len(y_true) != len(y_pred):
        raise ValueError(f"Alignment mismatch: {len(y_pred)} preds vs {len(y_true)} GT.")

    std_test = np.std(y_true, axis=0)
    mae_cd   = float(np.mean(np.abs(y_pred[:, 0] - y_true[:, 0])))
    mae_cl   = float(np.mean(np.abs(y_pred[:, 1] - y_true[:, 1])))
    r2       = r2_score(y_true, y_pred, multioutput='raw_values')
    return mae_cd / std_test[0], mae_cl / std_test[1], float(r2[0]), float(r2[1])


def _add_secondary_xaxis(ax, sensor_std_factor):
    """Add a top x-axis showing noise as % of the 4 selected sensors' mean std."""
    ax2 = ax.twiny()
    ax2.set_xlim(np.array(ax.get_xlim()) / sensor_std_factor)
    ax2.set_xlabel(r'Gaussian noise $\sigma$ (\% of selected sensors $\sigma$)',
                   fontsize='small')
    ax2.tick_params(axis='x', labelsize='small')
    return ax2


def plot_combined(clean_results, aug_slim_results, aug_full_results, output_path,
                  sensor_std_factor: float):
    """
    Three-way comparison:
      - Solid lines:        clean model degradation curve
      - Dashed lines:       slim encoder retrained with noise on CLEAN full model
      - Dash-dot lines:     slim encoder retrained on NOISE-TRAINED full model
    Points on the augmented curves are placed at their training alpha (matched eval).
    """
    c_a   = [r['noise_alpha_pct'] for r in clean_results]
    c_ncd = [r['norm_mae_cd']     for r in clean_results]
    c_ncl = [r['norm_mae_cl']     for r in clean_results]
    c_rcd = [r['r2_cd']           for r in clean_results]
    c_rcl = [r['r2_cl']           for r in clean_results]

    s_a   = [r['noise_alpha_pct'] for r in aug_slim_results]
    s_ncd = [r['norm_mae_cd']     for r in aug_slim_results]
    s_ncl = [r['norm_mae_cl']     for r in aug_slim_results]
    s_rcd = [r['r2_cd']           for r in aug_slim_results]
    s_rcl = [r['r2_cl']           for r in aug_slim_results]

    f_a   = [r['noise_alpha_pct'] for r in aug_full_results]
    f_ncd = [r['norm_mae_cd']     for r in aug_full_results]
    f_ncl = [r['norm_mae_cl']     for r in aug_full_results]
    f_rcd = [r['r2_cd']           for r in aug_full_results]
    f_rcl = [r['r2_cl']           for r in aug_full_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.6))

    kw = dict(markersize=5, linewidth=1.2)
    for ax, cd_y, cl_y, s_cd_y, s_cl_y, f_cd_y, f_cl_y, ylabel in [
        (ax1, c_ncd, c_ncl, s_ncd, s_ncl, f_ncd, f_ncl, r'Relative error ($L_1/\sigma$)'),
        (ax2, c_rcd, c_rcl, s_rcd, s_rcl, f_rcd, f_rcl, r'$R^2$'),
    ]:
        ax.plot(c_a, cd_y, 'o-',   color='tab:blue',   label=r'$C_d$ clean signal', **kw)
        ax.plot(c_a, cl_y, 's-',   color='tab:orange', label=r'$C_l$ clean signal', **kw)
        ax.plot(s_a, s_cd_y, 'o--',color='tab:blue',   label=r'$C_d$ slim train',   **kw)
        ax.plot(s_a, s_cl_y, 's--',color='tab:orange', label=r'$C_l$ slim train',   **kw)
        ax.plot(f_a, f_cd_y, 'o-.',color='tab:blue',   label=r'$C_d$ full train',   **kw)
        ax.plot(f_a, f_cl_y, 's-.',color='tab:orange', label=r'$C_l$ full train',   **kw)
        ax.set_xlabel(r'Gaussian noise $\sigma$ (\% of global $\sigma$)')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.spines['right'].set_visible(False)
        _add_secondary_xaxis(ax, sensor_std_factor)

    if ax2.get_ylim()[0] < 0:
        ax2.set_ylim(bottom=0.0)
    ax1.text(-0.15, 1.22, '(a)', transform=ax1.transAxes, size='large')
    ax2.text(-0.15, 1.22, '(b)', transform=ax2.transAxes, size='large')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=3,
               frameon=False,
               fontsize='small',
               handlelength=3.5,
               handletextpad=0.5,
               columnspacing=1.2)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(Path(output_path).with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(Path(output_path).with_suffix('.eps'), bbox_inches='tight')
    print(f"Saved combined plot to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng    = np.random.default_rng(SEED)
    rng_eval    = np.random.default_rng(SEEDeval)
    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')
    print(f"Using device: {device}")

    # --- Load base model config (for args, SHAP ranking, sensor range) ---
    base_ckp_dir, base_results_dir = find_model_paths(BASE_MODEL_ID, CHECKPOINTS_DIR)
    latent_file = next(
        f for f in sorted(base_results_dir.glob(f"{BASE_MODEL_ID}*latent_space.hdf5"))
        if '_eval' not in f.name
    )
    with h5py.File(latent_file, 'r') as hf:
        base_args    = Args(**json.loads(hf.attrs['args']))
        forces_mean  = hf['forces_mean'][:]
        forces_std   = hf['forces_std'][:]

    base_artifact_dir = (Path(RESULTS_DIR) / base_args.case /
                         "sensor_selection" / BASE_MODEL_ID)
    subset_eval_dir   = base_artifact_dir / "subset_evaluation"
    output_dir        = subset_eval_dir / "noise_augmented_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- SHAP ranking -> selected 4 sensor indices ---
    shap_path = base_artifact_dir / "mean_abs_shap_values.txt"
    mean_abs_shap    = np.loadtxt(shap_path)
    sensor_ranking   = np.argsort(mean_abs_shap)[::-1]
    selected_indices = np.sort(sensor_ranking[:NUM_SENSORS])
    print(f"Selected sensor indices: {selected_indices}")

    # --- Load and scale eval data ONCE ---
    loader_args_eval                    = deepcopy(base_args)
    project_root                        = Path(CHECKPOINTS_DIR).parent
    eval_paths                          = base_args.eval_dataset
    if isinstance(eval_paths, str):
        eval_paths = [eval_paths]
    loader_args_eval.datafile           = [str(project_root / p) for p in eval_paths]
    loader_args_eval.batch_size         = 512
    loader_args_eval.n_test             = 0
    loader_args_eval.augment_with_symmetry = False
    p_scaled_eval, control_scaled_eval, forces_scaled_eval, file_end_indices_eval = (
        prepare_raw_data(loader_args_eval))
    raw_forces_gt = load_raw_forces(loader_args_eval)

    sensor_std_factor = float(np.std(p_scaled_eval[:, selected_indices], axis=0).mean())
    print(f"Mean std of 4 selected sensors (normalised): {sensor_std_factor:.4f}")

    # --- Load and scale slim encoder training data ONCE ---
    loader_args_slim                    = deepcopy(base_args)
    loader_args_slim.batch_size         = SLIM_BATCH
    loader_args_slim.n_test             = 5000
    loader_args_slim.augment_with_symmetry = False
    p_scaled_slim, control_scaled_slim, forces_scaled_slim, file_end_indices_slim = (
        prepare_raw_data(loader_args_slim))

    # --- Main loop: one full model per training alpha ---
    aug_full_results = []

    for alpha_train in NOISE_ALPHAS_TRAIN:
        noise_sigma  = alpha_train   # data is z-scored; alpha is directly noise/signal ratio
        alpha_pct    = alpha_train * 100
        strdate_tag  = time.strftime("%Y%m%d_%H_%M_%S")

        # Attach noise config to args so it ends up in modelname / saved json
        train_args                       = deepcopy(base_args)
        train_args.sensor_noise_alpha_pct = alpha_pct   # stored for reference

        # Check if this full model was already trained
        # Match both old naming (noise2pct) and new naming (noise2.00pct)
        existing = (list(Path(CHECKPOINTS_DIR).glob(
                        f"**/*noise{alpha_pct:.2f}pct*_encoder.pth.tar")) or
                    list(Path(CHECKPOINTS_DIR).glob(
                        f"**/*noise{alpha_pct:.0f}pct*_encoder.pth.tar")))
        if existing:
            print(f"\nFound existing full model for alpha={alpha_pct:.2f}%, skipping training.")
            enc_ckpt  = existing[0]
            ckp_dir   = enc_ckpt.parent
            # Derive the force decoder path from the encoder path
            frc_ckpt  = Path(str(enc_ckpt).replace('_encoder.pth.tar', '_force.pth.tar'))
            # Reconstruct encoder + force_decoder from checkpoint
            encoder = TemporalEncoder(
                input_dim=mean_abs_shap.shape[0],
                latent_dim=base_args.latent_dim,
                hidden_dim=base_args.enc_hidden_dim,
                num_layers=base_args.n_layers).to(device)
            force_decoder = ForceDecoder(
                latent_dim=base_args.latent_dim,
                hidden_dim=base_args.forces_hidden_dim,
                arch=base_args.force_decoder_arch).to(device)
            load_checkpoint(encoder,       str(enc_ckpt), device)
            load_checkpoint(force_decoder, str(frc_ckpt), device)
            encoder.eval(); force_decoder.eval()
        else:
            encoder, _, force_decoder, ckp_dir, model_name = train_full_model_with_noise(
                train_args, device, noise_sigma, strdate_tag)

        # --- Train slim encoder on top of this noise-trained full model ---
        slim_ckpt = output_dir / f"slim_encoder_{NUM_SENSORS}sensors_fullnoise{alpha_pct:.2f}pct.pth.tar"
        slim_encoder = TemporalEncoder(
            input_dim=NUM_SENSORS,
            latent_dim=base_args.latent_dim,
            hidden_dim=base_args.enc_hidden_dim,
            num_layers=base_args.n_layers).to(device)

        if slim_ckpt.exists():
            print(f"  Found slim encoder checkpoint for alpha={alpha_pct:.2f}%, skipping training.")
        else:
            train_slim_encoder(slim_encoder, encoder,
                               p_scaled_slim, control_scaled_slim, forces_scaled_slim,
                               file_end_indices_slim, loader_args_slim,
                               selected_indices, noise_sigma,
                               device, slim_ckpt, rng)
        load_checkpoint(slim_encoder, str(slim_ckpt), device)
        slim_encoder.eval()

        # --- Evaluate at matching noise level ---
        print(f"  Evaluating at alpha={alpha_pct:.2f}%...")
        dl_eval_noisy, _ = build_dataloader_from_scaled(
            p_scaled_eval, control_scaled_eval, forces_scaled_eval,
            file_end_indices_eval, loader_args_eval, device,
            noise_sigma=noise_sigma, rng=rng_eval, shuffle_train=False)
        norm_mae_cd, norm_mae_cl, r2_cd, r2_cl = evaluate_force_prediction(
            slim_encoder, force_decoder, dl_eval_noisy,
            selected_indices, raw_forces_gt,
            forces_mean, forces_std, device)
        print(f"  norm_MAE: Cd={norm_mae_cd:.4f}, Cl={norm_mae_cl:.4f} | "
              f"R2: Cd={r2_cd:.4f}, Cl={r2_cl:.4f}")

        aug_full_results.append({
            'noise_alpha':     alpha_train,
            'noise_alpha_pct': alpha_pct,
            'noise_sigma':     noise_sigma,
            'norm_mae_cd':     norm_mae_cd,
            'norm_mae_cl':     norm_mae_cl,
            'r2_cd':           r2_cd,
            'r2_cl':           r2_cl,
        })

    # --- Save results ---
    results_file = output_dir / "noise_augmented_full_results.json"
    with open(results_file, 'w') as f:
        json.dump(aug_full_results, f, indent=4)
    print(f"\nSaved full-model augmented results to {results_file}")

    # --- Load existing clean-model and slim-augmented results for the combined plot ---
    clean_results_path    = subset_eval_dir / "noise_robustness_results.json"
    aug_slim_results_path = subset_eval_dir / "noise_augmented" / "noise_augmented_results.json"
    for p in [clean_results_path, aug_slim_results_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\n"
                "Run evaluate_noise_robustness.py and evaluate_noise_robustness_augmented.py first."
            )
    with open(clean_results_path)    as f: clean_results    = json.load(f)
    with open(aug_slim_results_path) as f: aug_slim_results = json.load(f)


    # --- Combined three-way plot ---
    combined_plot = subset_eval_dir / "noise_robustness_combined_full.png"
    plot_combined(clean_results, aug_slim_results, aug_full_results, combined_plot,
                  sensor_std_factor)

    print("\nDone.")


if __name__ == "__main__":
    main()