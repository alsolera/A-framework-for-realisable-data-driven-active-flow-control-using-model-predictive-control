import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=64, num_layers=2,
                 dropout_rate=0.0):
        """
        LSTM-based encoder that maps sequential input to a latent representation.

        Args:
            input_dim (int): Number of input features per time step (sensors + control).
            latent_dim (int): Dimension of the latent space.
            hidden_dim (int): Hidden state dimension of LSTM.
            num_layers (int): Number of LSTM layers.
            dropout_rate (float): Dropout rate for LSTM layers.
        """
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.output_norm = nn.LayerNorm(hidden_dim) # <<< Add LayerNorm for output hidden state
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, s_t):
        """
        Forward pass.

        Args:
            s_t (torch.Tensor): Input sequence of shape (batch, seq_length, features).

        Returns:
            z_t (torch.Tensor): Latent representation of shape (batch, latent_dim).
        """
        self.lstm.flatten_parameters()

        _, (h_n, _) = self.lstm(s_t)
        last_hidden = h_n[-1, :, :]  # Shape: (batch, hidden_dim)
        norm_hidden = self.output_norm(last_hidden)  # <<< Apply normalization
        z_t = self.fc(norm_hidden)  # <<< Pass normalized hidden state
        return z_t


class LatentDynamicsModel(nn.Module):
    def __init__(self, latent_dim=32, action_dim=2, hidden_dim=64, use_residual=True):
        """
        Latent dynamics model predicting z_t+1 given z_t and action a_t.

        Args:
            latent_dim (int): Dimension of the latent space.
            action_dim (int): Number of control actions.
            hidden_dim (int): Hidden layer dimension.
        """
        super(LatentDynamicsModel, self).__init__()
        self.use_residual = use_residual
        self.model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Optional normalization
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Optional normalization
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t, a_t):
        """
        Forward pass.

        Args:
            z_t (torch.Tensor): Current latent state of shape (batch, latent_dim).
            a_t (torch.Tensor): Control action of shape (batch, action_dim).

        Returns:
            z_t1 (torch.Tensor): Predicted latent state at t+1.
        """
        x = torch.cat([z_t, a_t], dim=-1)
        delta_z = self.model(x)
        if self.use_residual:
            z_t1 = z_t + delta_z # Additive residual connection
        else:
            z_t1 = delta_z # Original behavior (model predicts z_t1 directly)
        return z_t1


class ForceDecoder(nn.Module):
    def __init__(self, latent_dim=32, out_dim=2, hidden_dim=32, dropout_rate=0., arch="FCN"):
        """
        Decoder that maps the latent representation to aerodynamic force coefficients.
        Supports multiple architectures controlled by the 'arch' parameter.

        Args:
            latent_dim (int): Dimension of the latent space.
            out_dim (int): Dimension of the output (e.g., C_d, C_l).
            hidden_dim (int): Hidden layer dimension.
            dropout_rate (float): Dropout rate.
            arch (str): Architecture type, 'FCN' or 'ResNet'.
        """
        super(ForceDecoder, self).__init__()
        self.arch = arch

        if self.arch == "FCN":
            self.model = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, out_dim)
            )
        elif self.arch == "ResNet":
            # First layer to match dimensions if latent_dim != hidden_dim
            self.initial_layer = nn.Linear(latent_dim, hidden_dim)

            # First residual block
            self.block1 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim)
            )

            # Second residual block
            self.block2 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # Final output layer
            self.output_layer = nn.Linear(hidden_dim, out_dim)
        else:
            raise ValueError(f"Unknown ForceDecoder architecture: {self.arch}")

    def forward(self, z_t):
        """
        Predicts aerodynamic force coefficients from latent space.
        """
        if self.arch == "FCN":
            return self.model(z_t)
        elif self.arch == "ResNet":
            # Initial transformation
            x = self.initial_layer(z_t)
            # Pass through first residual block and add the input (skip connection)
            x = x + self.block1(x)
            # Pass through second residual block and add the input
            x = x + self.block2(x)
            # Final output
            return self.output_layer(x)


def compute_total_test_loss(
        z_pred, z_true, c_pred, c_true,
        lambda_mse=1.0, lambda_var=0.1, lambda_cov=0.01, lambda_cd_cl=1.0,
        return_components=False):
    """
    Computes total loss including:
    - MSE loss for next state prediction (z_pred vs. z_true)
    - VICReg-style variance and covariance regularization
    - Force coefficient prediction loss (c_pred vs. c_true)

    Args:
        z_pred (torch.Tensor): Predicted latent states. Shape: (batch_size, latent_dim)
        z_true (torch.Tensor): Ground truth latent states. Shape: (batch_size, latent_dim)
        c_pred (torch.Tensor): Predicted force coefficients (C_d, C_l). Shape: (batch_size, 2)
        c_true (torch.Tensor): True force coefficients. Shape: (batch_size, 2)
        lambda_mse (float): Weight for MSE loss.
        lambda_var (float): Weight for variance regularization.
        lambda_cov (float): Weight for covariance regularization.
        lambda_cd_cl (float): Weight for force coefficient loss.
        return_components (bool): If True, returns individual loss components.

    Returns:
        torch.Tensor: Total loss value.
    """
    batch_size, latent_dim = z_pred.shape

    # 1. MSE Loss (Predicting next latent state)
    mse_loss = torch.mean((z_pred - z_true) ** 2)

    # 2. Variance Regularization (Prevent collapse)
    z_pred_centered = z_pred - z_pred.mean(dim=0, keepdim=True)
    var = torch.var(z_pred_centered, dim=0)
    var_loss = torch.mean(torch.relu(1.0 - torch.sqrt(var + 1e-8)))  # Encourage variance near 1.0

    # 3. Covariance Regularization (Reduce redundancy)
    cov_matrix = (z_pred_centered.T @ z_pred_centered) / (batch_size - 1)
    # Create a mask for off-diagonal elements
    off_diag_mask = ~torch.eye(latent_dim, dtype=torch.bool, device=z_pred.device)
    # Calculate mean squared value of off-diagonal elements
    cov_loss = (cov_matrix[off_diag_mask] ** 2).mean()

    # 4. Force Coefficient Prediction Loss (Ensure latent space encodes aerodynamic information)
    cd_cl_loss = F.smooth_l1_loss(c_pred, c_true)

    # Total Loss
    total_loss = lambda_mse * mse_loss + lambda_var * var_loss + lambda_cov * cov_loss + lambda_cd_cl * cd_cl_loss

    if return_components:
        return total_loss, mse_loss.item(), var_loss.item(), cov_loss.item(), cd_cl_loss.item()
    return total_loss
