import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=size, padding=size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=size, padding=size//2)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ChessNNWithResiduals(nn.Module):
    def __init__(self):
        super(ChessNNWithResiduals, self).__init__()

        self.conv_init = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.res_block1 = ResidualBlock(channels=64, size=3)
        self.res_block2 = ResidualBlock(channels=64, size=5)
        self.res_block3 = ResidualBlock(channels=64, size=7)


        self.policy_head = nn.Sequential(
            nn.Linear(in_features=64*8*8, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1968),
        )


        self.value_head = nn.Sequential(
            nn.Linear(in_features=64*8*8, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = x.view(x.size(0), -1)

        policy = self.policy_head(x)

        value = self.value_head(x)

        return policy, value