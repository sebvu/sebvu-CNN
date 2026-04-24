import torch.nn as nn
import torch.optim as optim
import pandas as pd

class ClothingClassificationAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return # insert forward here

    def train(self, epochs, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

class Trainer:
    def __init__(self, epochs, learning_rate):
        self.model = ClothingClassificationAgent()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        self.epochs = epochs

    def train(self):
        for epoch in range(self.epochs):
            print(epoch)


    # insert more methods
    


def main():
    model = ClothingClassificationAgent()
    print(model)

    df = pd.read_csv("data/images.csv")
    print(df)


if __name__=="__main__":
    main()
