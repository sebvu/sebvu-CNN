import torch
import os

# hyprparameters
EPOCHS = 60
LEARNING_RATE = 0.0001
BATCH_SIZE=32
KERNEL_SIZE=3
EVAL_MODELS_PATH=os.path.join(os.getcwd(), "example-models") # change "models" to "example-models" to try the provided example models during eval

### DO NOT TOUCH BELOW ###

# global variables
DATA_CSV_PATH=os.path.join(os.getcwd(), "data", "images.csv")
MODELS_PATH=os.path.join(os.getcwd(), "models")

# hardware specific settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
