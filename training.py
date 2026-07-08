import torch
import torch.optim as optim
import numpy as np
import mlflow
import time
import os
import h5py
from pathlib import Path
from dataclasses import asdict
import json
import matplotlib
import argparse

matplotlib.use('Agg')  # Set non-interactive backend
from libs.models import TemporalEncoder, LatentDynamicsModel, ForceDecoder, compute_total_test_loss
from libs.data import get_prepared_data, loadData
from libs.test_encoder_predictor import test_encoder_predictor
from Plots.visualize_latent import visualize_latent_space
from parameters import Args
from copy import deepcopy


def save_latent_space(args, encoder, datafile, dataloader_no_shuffle, outdir, strdate, eval_run=False):
    """
    Saves the latent space representation of the dataset to an HDF5 file.
    Also stores the true values for comparison and the model configuration.
    """
    encoder.eval()
    latent_space_list = []
    device = next(encoder.parameters()).device

    # --- Collect latent space and corresponding sequences directly from the dataloader ---
    all_s, all_a, all_s1, all_C = [], [], [], []
    with torch.no_grad():
        for s_t_batch, a_seq_batch, s_t1_batch, c_seq_batch in dataloader_no_shuffle:
            s_t_batch_gpu = s_t_batch.to(device)
            z = encoder(s_t_batch_gpu)
            latent_space_list.append(z.cpu().numpy())

            # Also save the sequences from this batch
            all_s.append(s_t_batch.cpu().numpy())
            all_a.append(a_seq_batch.cpu().numpy())
            all_s1.append(s_t1_batch.cpu().numpy())
            all_C.append(c_seq_batch.cpu().numpy())

    latent_space = np.concatenate(latent_space_list, axis=0)
    s_t_all = np.concatenate(all_s, axis=0)
    a_seq_all = np.concatenate(all_a, axis=0)
    s_t1_all = np.concatenate(all_s1, axis=0)
    c_seq_all = np.concatenate(all_C, axis=0)

    # Align number of samples
    num_latent_samples = latent_space.shape[0]
    suffix = "_eval" if eval_run else ""
    # Use the modelname from args if it exists, otherwise build from strdate
    model_name_in_file = args.modelname if args.modelname else strdate
    fname = os.path.join(outdir, f"{model_name_in_file}{suffix}_latent_space.hdf5")
    print(f'Saving latent space and sequences to {fname}')

    with h5py.File(fname, 'w') as f:
        f.create_dataset('latent_space', data=latent_space)
        f.create_dataset('s_t_all', data=s_t_all)
        f.create_dataset('a_seq_all', data=a_seq_all)
        f.create_dataset('s_t1_all', data=s_t1_all)
        f.create_dataset('c_seq_all', data=c_seq_all)

        # Need to load scaling parameters to save them. We can do this without reloading all data.
        # We'll use the dataloader's args to ensure we're loading from the correct file list.
        temp_args_for_load = deepcopy(args)
        temp_args_for_load.datafile = datafile
        (_, _, _, p_mean, p_std, forces_mean, forces_std,
         control_mean, control_std, probes, _) = loadData(temp_args_for_load, printer=False)

        # Save scaling info and args for reproducibility
        f.create_dataset('p_mean', data=p_mean)
        f.create_dataset('p_std', data=p_std)
        f.create_dataset('forces_mean', data=forces_mean)
        f.create_dataset('forces_std', data=forces_std)
        f.create_dataset('control_mean', data=control_mean)
        f.create_dataset('control_std', data=control_std)
        f.create_dataset('probes', data=probes)
        f.attrs['args'] = json.dumps(asdict(args))

    return fname


# Parse command-line arguments first
parser = argparse.ArgumentParser(description="Train PLDM Model")
parser.add_argument("--comment", type=str, default="", help="A comment describing the training run.")

# Parse known args, allowing unknown args for Args dataclass or others
cli_args, unknown = parser.parse_known_args()

# Initialize default Args dataclass
args = Args()

# --- Query for comment if not provided and running interactively ---
if not cli_args.comment:
    try:
        cli_args.comment = input("Enter a comment for this training run (optional, press Enter to skip): ")
    except EOFError:  # Handle cases where input might not be available unexpectedly
        print("Could not read comment from input, proceeding without.")
        cli_args.comment = ""

# --- Add the comment to the args object ---
args.comment = cli_args.comment  # Store comment in the main args object

# seeding
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

if torch.cuda.is_available() and args.cuda:
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_properties(i).name}")
    device_idx = 2 if torch.cuda.device_count() > 2 else 0
    device = torch.device(f'cuda:{device_idx}')
else:
    device = torch.device('cpu')

print(f"Active device: {device}")

# Get case name
print(f'Case: {args.case}')

# creating directories if do not exist
Path(args.logdir).mkdir(parents=True, exist_ok=True)
Path(args.ckpdir).mkdir(parents=True, exist_ok=True)
Path(args.outdir).mkdir(parents=True, exist_ok=True)

# --- Dataloader for Training (from `args.datafile`) ---
dataloader_train, dataloader_test = get_prepared_data(args, device)

# --- Dataloader for Evaluation (from `args.eval_dataset`) ---
# Create a separate args object for the evaluation loader to avoid conflicts
eval_args = deepcopy(args)
eval_args.datafile = args.eval_dataset  # Use the eval dataset path
eval_args.n_test = 0  # Use all data from the eval file(s) for validation
eval_args.augment_with_symmetry = False  # Never augment the validation set
dataloader_eval, _ = get_prepared_data(eval_args, device, shuffle_train=False, shuffle_test=False)
print(f"Validation during training will use: {args.eval_dataset}")

# --- Dataloader for saving latent space (from training data, no shuffle) ---
dummy_args = deepcopy(args)
dummy_args.n_test = 0
dummy_args.augment_with_symmetry = False  # Ensure we save latent space for original data only
dataloader_no_shuffle, _ = get_prepared_data(dummy_args, device, shuffle_train=False)

# Initialize models
in_dim = dataloader_train.dataset[0][0].shape[-1]  # e.g. 90 sensor features
# Use the last dimension of the control input (which is 1), not the first dimension
action_dim = dataloader_train.dataset[0][1].shape[-1]
encoder = TemporalEncoder(input_dim=in_dim, latent_dim=args.latent_dim, hidden_dim=args.enc_hidden_dim,
                          num_layers=args.n_layers).to(device)
dynamics_model = LatentDynamicsModel(latent_dim=args.latent_dim, action_dim=action_dim, hidden_dim=args.dyn_hidden_dim,
                                     use_residual=args.residual_predictor).to(device)
force_decoder = ForceDecoder(latent_dim=args.latent_dim,
                             hidden_dim=args.forces_hidden_dim, dropout_rate=args.forces_dropout,
                             arch=args.force_decoder_arch).to(device)

# Optimizer
optimizer = optim.Adam(
    list(encoder.parameters()) + list(dynamics_model.parameters()) + list(force_decoder.parameters()),
    lr=args.lr,
)

strdate = time.strftime("%Y%m%d_%H_%M_%S")

args.modelname = (f'{strdate}_LSTM_dim{args.latent_dim}_lb{args.lookback}'
                  f'_l{args.n_layers}_h{args.dyn_hidden_dim}_dr{args.dropout}'
                  f'_lr{args.lr}_bs{args.batch_size}'
                  f'_wC_ntest{args.n_test}')
print(args.modelname)

# Set the MLflow tracking URI to your desired log directory
mlflow.set_tracking_uri(f"file://{str(Path(args.logdir).resolve())}")
mlflow.set_experiment(f"{args.case} LSTM")

print(f'execute: mlflow ui -p 5001 --backend-store-uri file://{os.getcwd()}/02_Logs/{args.case}')
print('Open in browser: http://localhost:5001/')

# Start MLflow run
with mlflow.start_run(run_name=args.modelname) as run:
    # Log only public attributes of the Args dataclass as hyperparameters
    public_args = {
        k: v
        for k, v in vars(args).items()
        if not k.startswith("_")
    }
    mlflow.log_params(public_args)

    # Log the comment as a tag (tags are good for short descriptions/categorization)
    if args.comment:
        mlflow.set_tag("run_comment", args.comment)
        print(f"MLflow Run Comment: {args.comment}")

    bestloss = float('inf')

    # Training loop
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        dynamics_model.train()
        force_decoder.train()

        # Initialize accumulators for epoch averages
        epoch_total_loss, epoch_mse_loss, epoch_var_loss, epoch_cov_loss, epoch_cd_cl_loss = 0, 0, 0, 0, 0
        dynamics_loss_count, force_loss_count = 0, 0  # Count how many times each loss is added

        num_batches_train = len(dataloader_train)

        for batch in dataloader_train:
            # Unpack batch: C_seq now contains future force coefficients
            # Shape: (batch, recursive_train_steps, 2)
            s_t, a_seq, s_t1, C_seq = batch  # <-- CHANGE: variable name C_seq
            s_t, a_seq, s_t1, C_seq = s_t.to(device), a_seq.to(device), s_t1.to(device), C_seq.to(device)

            # --- Add noise to control input for regularization ---
            if args.control_noise_std > 0.0 and encoder.training:  # Check for training mode
                noise = torch.randn_like(a_seq) * args.control_noise_std
                a_seq = a_seq + noise

            if args.recursive_train_steps == 1:
                # Action: (batch, 1, dim) -> (batch, dim)
                a_seq = a_seq[:, 0, :]
                # Force: (batch, 1, 2) -> (batch, 2)
                C_seq = C_seq[:, 0, :]

            # 1. Encode the past sensor sequence to obtain the initial latent state.
            z_t = encoder(s_t)  # shape: (batch, latent_dim)

            # 3. Recursively roll out the dynamics model using future control inputs.
            # Rollout and collect latent preds
            z_preds = [z_t]
            z = z_t
            if args.recursive_train_steps == 1:
                z = dynamics_model(z, a_seq)
                z_preds.append(z)
            else:
                for step in range(args.recursive_train_steps):
                    a_current = a_seq[:, step, :]
                    z = dynamics_model(z, a_current)
                    z_preds.append(z)

            # 3. Encode target states
            z_targets = [None] * len(z_preds)
            z_targets[0] = z_t  # Not used in loss directly, but for consistency
            for step in range(args.recursive_train_steps):
                # unsqueeze needed if encoder expects seq_len dimension
                target_input = s_t1[:, step, :]  # Shape: (batch, features)
                if target_input.ndim == 2:  # Add seq_len dim if missing
                    target_input = target_input.unsqueeze(1)
                z_targets[step + 1] = encoder(target_input)  # z_{t+step+1} target

            # 4. Calculate losses over the rollout steps
            batch_total_dynamics_loss = torch.tensor(0.0, device=device)
            batch_total_force_loss = torch.tensor(0.0, device=device)
            batch_mse_sum, batch_var_sum, batch_cov_sum, batch_cdcl_sum = 0, 0, 0, 0
            num_dynamics_steps_in_loss = 0
            num_force_steps_in_loss = 0

            # Loop through predicted states z_{t+1} to z_{t+k}
            for i, z_pred in enumerate(z_preds[1:], start=1):
                is_first = (i == 1)
                is_last = (i == args.recursive_train_steps)  # Corrected is_last check
                target = z_targets[i]

                apply_dynamics_loss_this_step = False

                # Check if dynamics loss applies based on loss_mode
                if args.loss_mode == "all" or \
                        (args.loss_mode == "first_and_last" and (is_first or is_last)) or \
                        (args.loss_mode == "last" and is_last):
                    apply_dynamics_loss_this_step = True

                # Apply force loss at every step
                apply_force_loss_this_step = True  # Apply force loss for all steps i=1..k

                # Calculate loss components IF they apply for this step
                mse_loss_i = torch.tensor(0.0, device=device)
                var_loss_i = torch.tensor(0.0, device=device)
                cov_loss_i = torch.tensor(0.0, device=device)
                cd_cl_loss_i = torch.tensor(0.0, device=device)

                # --- Dynamics & VICReg Loss Calculation ---
                if apply_dynamics_loss_this_step:
                    batch_size, latent_dim = z_pred.shape
                    mse_loss_i = torch.mean((z_pred - target) ** 2)

                    z_pred_centered = z_pred - z_pred.mean(dim=0, keepdim=True)
                    var_i = z_pred_centered.var(dim=0, unbiased=True)  # Use unbiased estimator
                    # Original VICReg paper uses std dev loss: sqrt(var + eps)
                    var_loss_i = torch.mean(torch.relu(1.0 - torch.sqrt(var_i + 1e-8)))

                    if batch_size > 1 and latent_dim > 1:
                        cov_matrix_i = (z_pred_centered.T @ z_pred_centered) / (batch_size - 1)
                        off_diag_mask_i = ~torch.eye(latent_dim, dtype=torch.bool, device=z_pred.device)
                        cov_loss_i = (cov_matrix_i[off_diag_mask_i] ** 2).mean()

                    # Accumulate weighted dynamics loss for this step
                    step_dynamics_loss = args.lambda_mse * mse_loss_i + \
                                         args.lambda_var * var_loss_i + \
                                         args.lambda_cov * cov_loss_i
                    batch_total_dynamics_loss += step_dynamics_loss
                    num_dynamics_steps_in_loss += 1

                    # Accumulate values for logging averages
                    batch_mse_sum += mse_loss_i.item()
                    batch_var_sum += var_loss_i.item()
                    batch_cov_sum += cov_loss_i.item()

                # --- Force Loss Calculation (using target C_seq for this step) ---
                if apply_force_loss_this_step:
                    # Target forces for the current step 'i' (prediction for t+i)
                    # C_seq has shape (batch, k, 2). Need target at index i-1.
                    if args.recursive_train_steps == 1:
                        c_true_i = C_seq  # Shape (batch, 2)
                    else:
                        c_true_i = C_seq[:, i - 1, :]  # Shape (batch, 2)

                    loss_fn = torch.nn.SmoothL1Loss()

                    # 1. Clean path for Encoder/Dynamics gradients
                    c_pred_clean = force_decoder(z_pred)
                    cd_cl_loss_primary = loss_fn(c_pred_clean, c_true_i)

                    # 2. Noisy, detached path for ForceDecoder regularization
                    cd_cl_loss_regularization = torch.tensor(0.0, device=device)
                    if args.force_decoder_noise_std > 0.0:
                        z_pred_noisy = z_pred.detach() + torch.randn_like(z_pred) * args.force_decoder_noise_std
                        c_pred_noisy = force_decoder(z_pred_noisy)
                        cd_cl_loss_regularization = loss_fn(c_pred_noisy, c_true_i)

                    # Combine the losses. The Encoder is guided by the primary loss.
                    # The Decoder is guided by both.
                    cd_cl_loss_i = cd_cl_loss_primary + cd_cl_loss_regularization

                    # Accumulate weighted force loss for this step
                    batch_total_force_loss += args.lambda_cd_cl * cd_cl_loss_i
                    num_force_steps_in_loss += 1

                    # Accumulate value for logging average
                    batch_cdcl_sum += cd_cl_loss_i.item()

            # 5. Calculate final batch loss
            # Average the accumulated losses by the number of steps they were computed over
            avg_batch_dynamics_loss = batch_total_dynamics_loss / num_dynamics_steps_in_loss if num_dynamics_steps_in_loss > 0 else torch.tensor(
                0.0, device=device)
            avg_batch_force_loss = batch_total_force_loss / num_force_steps_in_loss if num_force_steps_in_loss > 0 else torch.tensor(
                0.0, device=device)

            total_loss = avg_batch_dynamics_loss + avg_batch_force_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate epoch losses (using the final total loss for the batch)
            epoch_total_loss += total_loss.item()
            # Accumulate averages of components for logging
            if num_dynamics_steps_in_loss > 0:
                epoch_mse_loss += batch_mse_sum / num_dynamics_steps_in_loss
                epoch_var_loss += batch_var_sum / num_dynamics_steps_in_loss
                epoch_cov_loss += batch_cov_sum / num_dynamics_steps_in_loss
                dynamics_loss_count += 1  # Count batches where dynamics loss was > 0
            if num_force_steps_in_loss > 0:
                epoch_cd_cl_loss += batch_cdcl_sum / num_force_steps_in_loss
                force_loss_count += 1  # Count batches where force loss was > 0

        # Compute epoch-wise mean and log
        avg_epoch_total_loss = epoch_total_loss / num_batches_train
        avg_epoch_mse_loss = epoch_mse_loss / dynamics_loss_count if dynamics_loss_count > 0 else 0
        avg_epoch_var_loss = epoch_var_loss / dynamics_loss_count if dynamics_loss_count > 0 else 0
        avg_epoch_cov_loss = epoch_cov_loss / dynamics_loss_count if dynamics_loss_count > 0 else 0
        avg_epoch_cd_cl_loss = epoch_cd_cl_loss / force_loss_count if force_loss_count > 0 else 0

        mlflow.log_metric("train_loss", avg_epoch_total_loss, step=epoch)
        mlflow.log_metric("train_mse_loss", avg_epoch_mse_loss, step=epoch)
        mlflow.log_metric("train_var_loss", avg_epoch_var_loss, step=epoch)
        mlflow.log_metric("train_cov_loss", avg_epoch_cov_loss, step=epoch)
        mlflow.log_metric("train_cd_cl_loss", avg_epoch_cd_cl_loss, step=epoch)

        # Validation loop
        with torch.no_grad():
            encoder.eval()
            dynamics_model.eval()
            force_decoder.eval()  # Set force decoder to eval mode

            num_batches_test = len(dataloader_eval)  # Use the dedicated eval loader
            test_loss = 0
            test_mse, test_var, test_cov, test_cd_cl = 0, 0, 0, 0  # Reset test accumulators

            for batch in dataloader_eval:  # Use the dedicated eval loader
                # Unpack validation batch (force target is sequence, but we only use final)
                s_t, a_seq, s_t1, C_seq = batch  # <-- Name change C_seq
                s_t, a_seq, s_t1, C_seq = s_t.to(device), a_seq.to(device), s_t1.to(device), C_seq.to(device)

                # Get the FINAL force target C_t for validation comparison
                if args.recursive_train_steps == 1:
                    a_seq = a_seq[:, 0, :]
                    C_t = C_seq[:, 0, :]  # Shape (batch, 2)
                else:
                    C_t = C_seq[:, -1, :]  # Shape (batch, 2) - Target force at the end of the window

                # --- Rollout ---
                z_t = encoder(s_t)
                # Compute target latent state z_t1_real from the final future sensor reading
                target_input_val = s_t1[:, -1, :]  # Final sensor state
                if target_input_val.ndim == 2:
                    target_input_val = target_input_val.unsqueeze(1)
                z_t1_real = encoder(target_input_val)

                # Predict final state z_pred after k steps
                z_pred = z_t
                if args.recursive_train_steps == 1:
                    z_pred = dynamics_model(z_t, a_seq)
                else:
                    for step in range(args.recursive_train_steps):
                        a_current = a_seq[:, step, :]
                        z_pred = dynamics_model(z_pred, a_current)
                # z_pred is now the prediction for z_{t+k}

                # --- Decode force from FINAL predicted state ---
                c_pred = force_decoder(z_pred)

                # --- Compute loss using FINAL states/targets ---
                val_total, val_mse, val_var, val_cov, val_cdcl = compute_total_test_loss(
                    z_pred, z_t1_real, c_pred, C_t,  # Compare final pred vs final targets
                    args.lambda_mse, args.lambda_var,
                    args.lambda_cov, args.lambda_cd_cl, return_components=True
                )

                test_loss += val_total.item()  # Use item() here
                test_mse += val_mse  # Already float items
                test_var += val_var
                test_cov += val_cov
                test_cd_cl += val_cdcl

            # Calculate average validation losses
            loss_test = test_loss / num_batches_test
            loss_mse_test = test_mse / num_batches_test
            loss_var_test = test_var / num_batches_test
            loss_cov_test = test_cov / num_batches_test
            loss_cd_cl_test = test_cd_cl / num_batches_test

            mlflow.log_metric("test_loss", loss_test, step=epoch)
            mlflow.log_metric("test_mse_loss", loss_mse_test, step=epoch)
            mlflow.log_metric("test_var_loss", loss_var_test, step=epoch)
            mlflow.log_metric("test_cov_loss", loss_cov_test, step=epoch)
            mlflow.log_metric("test_cd_cl_loss", loss_cd_cl_test, step=epoch)

        # Save best model
        if loss_test < bestloss and epoch > (args.epochs * 0.9 - 1):
            bestloss = loss_test
            checkpoint_encoder = {'state_dict': encoder.state_dict()}
            ckp_file_encoder = f'{args.ckpdir}{args.modelname}_encoder.pth.tar'
            torch.save(checkpoint_encoder, ckp_file_encoder)

            checkpoint_dynamics = {'state_dict': dynamics_model.state_dict()}
            ckp_file_dynamics = f'{args.ckpdir}{args.modelname}_dynamics.pth.tar'
            torch.save(checkpoint_dynamics, ckp_file_dynamics)

            checkpoint_force = {'state_dict': force_decoder.state_dict()}
            ckp_file_force = f'{args.ckpdir}{args.modelname}_force.pth.tar'
            torch.save(checkpoint_force, ckp_file_force)
            # Log the checkpoint as an artifact

            # --- Save the args dataclass as a json file ---
            args_file_path = f'{args.ckpdir}{args.modelname}_args.json'
            with open(args_file_path, 'w') as f:
                json.dump(asdict(args), f, indent=4)

            mlflow.log_artifact(ckp_file_encoder, artifact_path="checkpoints")
            mlflow.log_artifact(ckp_file_dynamics, artifact_path="checkpoints")

        print(
            f"Epoch {epoch}/{args.epochs} - Loss: {avg_epoch_total_loss:.6f} -  Test Loss: {loss_test:.6f} - Best Loss: {bestloss:.6f}")

    # Save latent space for the whole dataset
    fname = save_latent_space(args, encoder, args.datafile, dataloader_no_shuffle, args.outdir, strdate)
    mlflow.log_artifact(fname, artifact_path="latent_space")
    output_plot_path_pca, axis_lims, cd_lims, cl_lims, fitted_pca = visualize_latent_space(fname,
                                                                                           strdate,
                                                                                           args.outdir)
    mlflow.log_artifact(output_plot_path_pca, artifact_path="PCA_plots")
    mlflow.log_artifact(Path(output_plot_path_pca).with_suffix('.pdf'), artifact_path="PCA_plots_pdf")

    # Save latent space for the eval dataset
    dummy_args.datafile = args.eval_dataset
    dataloader_eval, _ = get_prepared_data(dummy_args, device, shuffle_train=False)
    fname = save_latent_space(args, encoder, args.eval_dataset, dataloader_eval, args.outdir, strdate, eval_run=True)
    mlflow.log_artifact(fname, artifact_path="latent_space_eval")

    output_plot_path_pca_eval, _, _, _, _ = visualize_latent_space(fname, strdate, args.outdir,
                                                                suffix='_eval',
                                                                axis_limits=axis_lims,
                                                                c_limits_cd=cd_lims,
                                                                c_limits_cl=cl_lims,
                                                                pca_object=fitted_pca)
    mlflow.log_artifact(output_plot_path_pca_eval, artifact_path="PCA_plots_eval")
    mlflow.log_artifact(Path(output_plot_path_pca_eval).with_suffix('.pdf'), artifact_path="PCA_plots_eval_pdf")

    # Test model
    error_plot_path, forces_error_plot_path, pred_sample_plot_path, decoder_eval_plot_path, scatter_plot_path = test_encoder_predictor(
        strdate, args)
    mlflow.log_artifact(error_plot_path, artifact_path="error_plot")
    mlflow.log_artifact(forces_error_plot_path, artifact_path="forces_error_plot")
    mlflow.log_artifact(pred_sample_plot_path, artifact_path="pred_sample_plot")
    mlflow.log_artifact(decoder_eval_plot_path, artifact_path="decoder_eval_plot")
    mlflow.log_artifact(scatter_plot_path, artifact_path="scatter_plot_path")