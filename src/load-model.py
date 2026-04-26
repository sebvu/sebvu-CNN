import torch
import os
import pandas as pd
from dataInterpretation import interpretResults
from ClothingClassificationAgent import ClothingClassificationAgent
from globalVariables import EPOCHS, LEARNING_RATE, DEVICE, DATA_CSV_PATH, BATCH_SIZE, EVAL_MODELS_PATH
from ClothingDataset import ClothingDataset
from torch.utils.data import DataLoader


def main():
    """
        used to evaluate existing models
    """
    dirs = {
            index: label
            for index, label in enumerate(os.listdir(EVAL_MODELS_PATH))
    }
    modelID = -1 # default
    dirsLen = int(len(dirs))
    displayText = "------\nSelect a model ID to eval.\n"

    while modelID < 0 or modelID > dirsLen - 1:
        print(displayText)

        for index in range(dirsLen):
            print(f"{index}: {dirs[index]}")

        modelID = int(input("model ID: "))

        if modelID < 0 or modelID > dirsLen - 1:
            displayText = f"------\n{modelID} is not a valid ID. Try again.\n"

    ### model successfully selected ###

    model_dir_name = dirs[modelID]
    dir_path = os.path.join(os.getcwd(), "models", model_dir_name)
    arch_path = os.path.join(dir_path, f"{model_dir_name}_architecture.txt")
    model_path = os.path.join(dir_path, f"{model_dir_name}_model.pt")

    # extract architecture
    with open(arch_path) as f:
        ops = f.readline().split(" ")
        kernel_size, is_deep = int(ops[1]), ops[3] == "True"

    """
        note that EPOCHS and LEARNING_RATE are just fillers, there will be no training involved
        kernel_size and is_deep is necessary
    """
    agent = ClothingClassificationAgent(EPOCHS, LEARNING_RATE, kernel_size, is_deep=is_deep, device=DEVICE)
    agent.model.load_state_dict(torch.load(model_path, weights_only=True))

    """
        manually just get the test data
    """

    df = pd.read_csv(DATA_CSV_PATH) # training data
    # df = df[df["label"] != "Not sure"] # filter out the "Not sure" label
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle (REMOVED GOING FORWARD)

    n = len(df) # split 70% train, 15% validation, 15% test
    test_df = df.iloc[int(0.85 * n):]

    labels = { # create the labels associated w/ids
            label: index
            for index, label in enumerate(dict.fromkeys(df["label"]))
    }

    test_dataset = ClothingDataset(test_df, labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    interpretResults(agent, test_dataset, test_loader, test_name=model_dir_name, save_only_if_better=False, display_confusion_matrix=True) # must be false to avoid overwriting



if __name__=="__main__":
    main()
