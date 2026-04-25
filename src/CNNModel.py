import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, kernel_size, is_deep):
        super().__init__()

        self.is_deep = is_deep
        self.k_size = kernel_size
        self.padding = self.k_size // 2 # general rule to preserve correct padding

        if not self.is_deep:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=self.k_size, padding=self.padding)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=self.k_size, padding=self.padding)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 32 * 32, 128)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, 10)
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=self.k_size, padding=self.padding)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=self.k_size, padding=self.padding)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=self.k_size, padding=self.padding)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=self.k_size, padding=self.padding)
            self.bn4 = nn.BatchNorm2d(256)
            self.pool = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256 * 8 * 8, 128)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        # softmax is implicitly activated on the output via CrossEntropyLoss
        
        if not self.is_deep:
            x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
            x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
            x = self.flatten(x)
            x = self.dropout(nn.functional.relu(self.fc1(x)))
        else:
            x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
            x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
            x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
            x = self.pool(nn.functional.relu(self.bn4(self.conv4(x))))
            x = self.flatten(x)
            x = self.dropout(nn.functional.relu(self.fc1(x)))
        return self.fc2(x)
