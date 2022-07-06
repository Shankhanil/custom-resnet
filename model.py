#@title
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
import torch.optim as optim

from torchsummary import summary
# summary(net, (3,32,32))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
epoch = 30
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=epoch)