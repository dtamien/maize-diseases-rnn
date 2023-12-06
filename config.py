log_level = "INFO"

# Specificities of the project
device = "mps"
data_dir = "data/"

# Learning checkpoints
use_checkpoint = True
with_lrs_checkpoint_path = "checkpoints/with_lrs/epoch_10.pth"
without_lrs_checkpoint_path = "checkpoints/without_lrs/epoch_10.pth"
with_lrs_checkpoint_save_dir = "checkpoints/with_lrs"
without_lrs_checkpoint_save_dir = "checkpoints/without_lrs"

# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
