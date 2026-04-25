import pandas as pd
from ClothingDataset import ClothingDataset
from torch.utils.data import DataLoader
from ClothingClassificationAgent import ClothingClassificationAgent
from dataVisualization import displayResults

def main():
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
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # hyprparameters
    epochs = 20
    learning_rate = 0.0001

    # no augmentation
    train_dataset_noaug = ClothingDataset(train_df, labels)
    train_loader_noaug = DataLoader(train_dataset_noaug, batch_size=32, shuffle=True)

    # not_deep 
    CC_not_deep = ClothingClassificationAgent(epochs, learning_rate) # instantiate model trainer, and train
    CC_not_deep.train(train_loader_noaug, val_loader)
    true_labels_not_deep, predictions_not_deep = CC_not_deep.evaluate(test_loader)
    displayResults(true_labels_not_deep, predictions_not_deep, test_dataset)

    # deep
    CC_deep = ClothingClassificationAgent(epochs, learning_rate, is_deep=True)
    CC_deep.train(train_loader_noaug, val_loader)
    true_labels_deep, predictions_deep = CC_deep.evaluate(test_loader)
    displayResults(true_labels_deep, predictions_deep, test_dataset)

    # augmentation
    train_dataset_aug = ClothingDataset(train_df, labels, True)
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=32, shuffle=True)

    # not_deep 
    CC_not_deep = ClothingClassificationAgent(epochs, learning_rate) # instantiate model trainer, and train
    CC_not_deep.train(train_loader_aug, val_loader)
    true_labels_not_deep, predictions_not_deep = CC_not_deep.evaluate(test_loader)
    displayResults(true_labels_not_deep, predictions_not_deep, test_dataset)

    # deep
    CC_deep = ClothingClassificationAgent(epochs, learning_rate, is_deep=True)
    CC_deep.train(train_loader_aug, val_loader)
    true_labels_deep, predictions_deep = CC_deep.evaluate(test_loader)
    displayResults(true_labels_deep, predictions_deep, test_dataset)

if __name__=="__main__":
    main()
