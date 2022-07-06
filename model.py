<<<<<<< HEAD
import torch.optim as optim
import torch.nn as nn
import torch
import config 
import dataset

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

=======
#@title
>>>>>>> cde55c924c6ebb2fc04925879042a9f6359c1d66
class Net(nn.Module):
    def __init__(self, in_channels, resblock, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(256, 256, downsample=True),

            resblock(256, 256, downsample=False)
        )


        # self.layer4 = nn.Sequential(
        #     resblock(256, 512, downsample=True),
        #     resblock(512, 512, downsample=False)
        # )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(1024, 10)

    def forward(self, input):
        _input = input
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        # input = self.layer4(input)
        # input = self.gap(input)
        input = torch.flatten(input,1)
        input = self.fc(input)

        # print(input.shape)

        return input
  
net = Net(3, ResBlock, outputs = 4)
<<<<<<< HEAD

# from torchsummary import summary
=======
import torch.optim as optim

from torchsummary import summary
>>>>>>> cde55c924c6ebb2fc04925879042a9f6359c1d66
# summary(net, (3,32,32))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
<<<<<<< HEAD
# epoch = 30
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataset.trainloader), epochs=config.epoch)
=======
epoch = 30
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=epoch)
>>>>>>> cde55c924c6ebb2fc04925879042a9f6359c1d66
