import pandas as pd
from ClothingDataset import ClothingDataset
from torch.utils.data import DataLoader
from ClothingClassificationAgent import ClothingClassificationAgent
from dataInterpretation import interpretResults
from globalVariables import (
        DATA_CSV_PATH, 
        MODELS_PATH,

        # hyprparameters (refer to globalVariables.py
        EPOCHS,
        LEARNING_RATE,
        BATCH_SIZE,
        KERNEL_SIZE,
        DEVICE
        )

def main():
    """
        used to train the actual models
    """

    # what's the training device?!
    print(f"using device: {DEVICE}")

    df = pd.read_csv(DATA_CSV_PATH) # training data
    # df = df[df["label"] != "Not sure"] # filter out the "Not sure" label
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle

    labels = { # create the labels associated w/ids
            label: index
            for index, label in enumerate(dict.fromkeys(df["label"]))
    }

    n = len(df) # split 70% train, 15% validation, 15% test
    train_df = df.iloc[:int(0.7 * n)]
    val_df = df.iloc[int(0.7 * n):int(0.85 * n)]
    test_df = df.iloc[int(0.85 * n):]

    val_dataset = ClothingDataset(val_df, labels)
    test_dataset = ClothingDataset(test_df, labels)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # no augmentation
    train_dataset_noaug = ClothingDataset(train_df, labels, is_augmented=False)
    train_loader_noaug = DataLoader(train_dataset_noaug, batch_size=BATCH_SIZE, shuffle=True)

    # # not_deep_noaug
    CC_not_deep_noaug = ClothingClassificationAgent(EPOCHS, LEARNING_RATE, KERNEL_SIZE, is_deep=False, device=DEVICE) 
    CC_not_deep_noaug.train(train_loader_noaug, val_loader, save_path=f"{MODELS_PATH}not_deep_noaug/")
    interpretResults(CC_not_deep_noaug, test_dataset, test_loader, test_name="not_deep_noaug", save_only_if_better=True)

    # deep_noaug
    CC_deep_noaug = ClothingClassificationAgent(EPOCHS, LEARNING_RATE, KERNEL_SIZE, is_deep=True, device=DEVICE)
    CC_deep_noaug.train(train_loader_noaug, val_loader, save_path=f"{MODELS_PATH}deep_noaug/")
    interpretResults(CC_deep_noaug, test_dataset, test_loader, test_name="deep_noaug", save_only_if_better=True)

    # augmentation
    train_dataset_aug = ClothingDataset(train_df, labels, is_augmented=True)
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True)

    # not_deep_aug
    CC_not_deep_aug = ClothingClassificationAgent(EPOCHS, LEARNING_RATE, KERNEL_SIZE, is_deep=False, device=DEVICE)
    CC_not_deep_aug.train(train_loader_aug, val_loader, save_path=f"{MODELS_PATH}not_deep_aug/")
    interpretResults(CC_not_deep_aug, test_dataset, test_loader, test_name="not_deep_aug", save_only_if_better=True)

    # deep_aug
    CC_deep_aug = ClothingClassificationAgent(EPOCHS, LEARNING_RATE, KERNEL_SIZE, is_deep=True, device=DEVICE)
    CC_deep_aug.train(train_loader_aug, val_loader, save_path=f"{MODELS_PATH}deep_aug/")
    interpretResults(CC_deep_aug, test_dataset, test_loader, test_name="deep_aug", save_only_if_better=True)

if __name__=="__main__":
    main()
