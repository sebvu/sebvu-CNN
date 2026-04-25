import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ClothingClassificationAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # softmax is implicitly activated on the output via CrossEntropyLoss
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        return self.fc2(x)

class ClothingDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.labels = {
                label: index
                for index, label in enumerate(dict.fromkeys(df["label"]))
        }
        self.transform = transforms.Compose([
            # resize all images to 128x127, dataset images may have varied resolutions and the networks expects a fixed resolution
            transforms.Resize((128, 128)), 
            # covert PIL image to python tensor shape of [3, 128, 128]. rescale pixel values from 0-255 int to 0-1 floats
            # channel ordered as [R, G, B]
            transforms.ToTensor(), 
            # normalize using mean=0.5, std=0.5
            # zero-centered -1 > 1 data trains more stably
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        row_image_id = row["image"]
        row_label = row["label"]
        img = Image.open(f"data/images/{row_image_id}.jpg")

        img_normalized_tensor = self.transform(img)

        return (img_normalized_tensor, self.labels[row_label])

class Trainer:
    def __init__(self, epochs: int, learning_rate: float):
        self.model = ClothingClassificationAgent()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = epochs

    def train(self, dataloader: DataLoader) -> None:
        for epoch in range(self.epochs):
            for X_batch, y_batch in dataloader:
                # X_batch is the 4D input of the image tensors that were transformed
                # y_batch is the answer key!

                # forward pass, produces a prediction
                y_hat = self.model(X_batch)

                # calculating the loss through softmax and comparing how much prob you assigned to wrong vs right
                # all through CrossEntropyLoss
                # comparing y_hat with the 'true' value of y_batch
                loss = self.loss_fn(y_hat, y_batch)

                # zero the gradients to prevent gradients from stacking on top of each other self.optimizer.zero_grad(DataLoadeDataLoaderr)
                self.optimizer.zero_grad()

                # walks backward through every operation that produced that loss
                # calculates a gradient for every single param
                loss.backward()

                # looks at the gradients loss.backward() produces, then shoves
                # the parameters to the direction of least loss
                with torch.no_grad():
                    self.optimizer.step()

                print(f"epoch: {epoch}, loss: {loss}")

def main():
    df = pd.read_csv("data/images.csv")

    dataset = ClothingDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    trainer = Trainer(20, 0.001)
    trainer.train(dataloader)



if __name__=="__main__":
    main()
