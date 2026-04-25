import torch

# global variables

DATA_CSV_PATH="data/images.csv"
MODELS_PATH="models/"

# hyprparameters

EPOCHS = 3
LEARNING_RATE = 0.0001
BATCH_SIZE=32
KERNEL_SIZE=3

# hardware specific settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
