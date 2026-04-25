import pandas as pd
from ClothingDataset import ClothingDataset
from torch.utils.data import DataLoader
from ClothingClassificationAgent import ClothingClassificationAgent
from dataVisualization import displayResults

def main():
    # hyprparameters
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    BATCH_SIZE=32
    KERNEL_SIZE=3
    ###

    df = pd.read_csv("data/images.csv") # training data
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

    # not_deep_noaug
    CC_not_deep_noaug = ClothingClassificationAgent(EPOCHS, LEARNING_RATE, KERNEL_SIZE, is_deep=False) 
    CC_not_deep_noaug.train(train_loader_noaug, val_loader)
    true_labels_not_deep, predictions_not_deep = CC_not_deep_noaug.evaluate(test_loader)
    displayResults(true_labels_not_deep, predictions_not_deep, test_dataset)

    # deep_noaug
    CC_deep_noaug = ClothingClassificationAgent(EPOCHS, LEARNING_RATE, KERNEL_SIZE, is_deep=True)
    CC_deep_noaug.train(train_loader_noaug, val_loader)
    true_labels_deep, predictions_deep = CC_deep_noaug.evaluate(test_loader)
    displayResults(true_labels_deep, predictions_deep, test_dataset)

    # augmentation
    train_dataset_aug = ClothingDataset(train_df, labels, is_augmented=True)
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True)

    # not_deep_aug
    CC_not_deep_aug = ClothingClassificationAgent(EPOCHS, LEARNING_RATE, KERNEL_SIZE, is_deep=False)
    CC_not_deep_aug.train(train_loader_aug, val_loader)
    true_labels_not_deep, predictions_not_deep = CC_not_deep_aug.evaluate(test_loader)
    displayResults(true_labels_not_deep, predictions_not_deep, test_dataset)

    # deep_aug
    CC_deep_aug = ClothingClassificationAgent(EPOCHS, LEARNING_RATE, KERNEL_SIZE, is_deep=True)
    CC_deep_aug.train(train_loader_aug, val_loader)
    true_labels_deep, predictions_deep = CC_deep_aug.evaluate(test_loader)
    displayResults(true_labels_deep, predictions_deep, test_dataset)

if __name__=="__main__":
    main()
