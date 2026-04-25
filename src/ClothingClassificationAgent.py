import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from CNNModel import CNNModel

class ClothingClassificationAgent:
    def __init__(self, epochs: int, learning_rate: float, kernel_size, is_deep: bool = False):
        self.model = CNNModel(kernel_size, is_deep)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = epochs
        # within 5 epochs if improvement not found, reduce steps by half
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
                )

    def evaluate(self, dataloader):
        self.model.eval() # turn off dropout and batchnorm training behavior
        all_preds = [] # predictions 
        all_labels = []

        with torch.no_grad(): # no gradient tracking
            for X_batch, y_batch in dataloader:
                y_hat = self.model(X_batch)
                preds = y_hat.argmax(dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.tolist())

        return all_labels, all_preds


    def train(self, dataloader: DataLoader, val_loader: DataLoader) -> None:
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
                self.optimizer.step()

            # after all batches, check validation loss
            val_labels, val_preds = self.evaluate(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            self.scheduler.step(val_acc) # scheduler steps w/new val_acc info
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"epoch {epoch+1}/{self.epochs} — val_acc: {val_acc:.3f} — lr: {current_lr:.6f}")

            self.model.train()  # switch back to train mode after evaluate
