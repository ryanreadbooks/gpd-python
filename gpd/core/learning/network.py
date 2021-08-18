import torch.nn as nn
import torch.nn.functional as F

class GPDClassifier(nn.Module):
    """
    Input: (batch_size, input_chann, 60, 60)
    """
    def __init__(self, input_chann, dropout=False):
        super(GPDClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_chann, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 32, 512)
        self.dp = nn.Dropout2d(p=0.5, inplace=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.if_dropout = dropout

    def forward(self, x):
        x = self.pool(self.conv1(x))
        # print(x.shape)
        x = self.pool(self.conv2(x))
        # print(x.shape)
        x = self.pool(self.conv3(x))
        # print(x.shape)
        x = x.view(-1, 7 * 7 * 32)
        x = self.relu(self.fc1(x))
        if self.if_dropout:
            x = self.dp(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

if __name__ == '__main__':
    import torch
    x = torch.rand(4, 12, 60, 60)
    cls = GPDClassifier(input_chann=12, dropout=True)
    y = cls(x)
    print(y)