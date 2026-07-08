from dataclasses import dataclass, field
from typing import List
import git


@dataclass
class Args:
    # Traceability information
    commit_sha: str = git.Repo(search_parent_directories=True).head.object.hexsha
    """the current git commit hash"""

    # Deterministic training
    seed: int = 1234
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    # Model and training parameters
    datafile: List[str] = field(default_factory=lambda: ['01_Data/jet_2Dtruck_20250307_FMsignal_50000.hdf5',
                                                         '01_Data/jet_2Dtruck_20250831_noAction_2500.hdf5'])
    """path to the data file(s)"""
    eval_dataset: List[str] = field(default_factory=lambda: ['01_Data/jet_2Dtruck_01092024_chirp15u_9999.hdf5'])
    """path to the evaluation dataset(s)"""
    modeltype: str = 'LSTM'
    """name of the model"""
    lookback: int = 32
    """lookback for the model"""
    recursive_train_steps: int = 5
    """recursive_train_steps for the predictor training"""
    residual_predictor: bool = True
    """Learn increment instead of next value in predictor model"""
    loss_mode: str = "all"
    """recursive train loss points for the predictor training: last, first_and_last, all"""
    augment_with_symmetry: bool = True
    """if toggled, data will be augmented with symmetric counterparts"""
    n_layers: int = 1
    """number of layers in the model"""
    enc_hidden_dim: int = 64
    """dimension of encoder hidden layers"""
    dyn_hidden_dim: int = 256
    """dimension of dynamics model hidden layers"""
    forces_hidden_dim: int = 128
    """dimension of forces decoder model hidden layers"""
    force_decoder_arch: str = "ResNet"
    """Architecture for the force decoder: 'FCN' or 'ResNet'"""
    force_decoder_noise_std: float = 0.0
    """Standard deviation of Gaussian noise to add to latent states fed to the ForceDecoder for regularization."""
    control_noise_std: float = 0.0  # Std dev of Gaussian noise to add to control inputs during training
    """Standard deviation of Gaussian noise to add to control inputs during training"""
    forces_dropout: float = 0.25
    """dropout in forces decoder model"""
    latent_dim: int = 8
    """dimension of latent space"""
    dropout: float = 0.0
    """dropout rate"""
    batch_size: int = 512
    """batch size"""
    lr: float = 1e-3
    """learning rate"""
    epochs: int = 300
    """number of epochs"""
    n_test: int = 0
    """number of test samples"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    DATA_TO_GPU: bool = cuda and False
    """whether to move data to GPU"""
    lambda_mse: float = 1.0
    """Weight for MSE loss. Usually 1 as reference"""
    lambda_var: float = 0.02
    """Weight for variance regularization."""
    lambda_cov: float = 0.02
    """Weight for covariance regularization."""
    lambda_cd_cl: float = 0.25
    """Weight for force coefficient prediction loss."""
    comment: str = ""
    """Optional user comment describing the run, logged as MLflow tag"""

    # Generated paths and names are now initialized after the object is created
    case: str = ''
    """case name (generated)"""
    logdir: str = ''
    """log dir (generated)"""
    ckpdir: str = ''
    """checkpoint files dir (generated)"""
    outdir: str = ''
    """results files dir (generated)"""
    modelname: str = ''
    """model name, to be filled in runtime"""

    def __post_init__(self):
        """This method is called after the dataclass is initialized."""
        # Now we can safely access self.datafile because the instance exists
        if self.datafile and isinstance(self.datafile, list):
            # Base the case name on the first file in the list
            self.case = self.datafile[0].split('/')[-1].split('.hdf5')[0].split('.csv')[0].split('_no_fields')[0]
            self.logdir = f'02_Logs/{self.case}/'
            self.ckpdir = f'03_Checkpoints/{self.case}/'
            self.outdir = f'04_Results/{self.case}/'

