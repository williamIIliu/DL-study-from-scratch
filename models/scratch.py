import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.cnn2= nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3, stride=2),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self,x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        return x

if __name__ == '__main__':
    net = Conv()
    print(net)