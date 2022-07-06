import torch
import torch.nn as nn
import torchvision.transforms as transforms
import logging
import torch.optim as optim
from model import ResBlock, Net
from dataset import trainset, trainloader, testset, testloader
import config
import os

# setting up logging
logging.basicConfig(filename = config.PATH["LOGS"])


net = Net(3, ResBlock, outputs = 4)

# summary(net, (3,32,32))
pytorch_total_params = sum(p.numel() for p in net.parameters())
logging.info(f"total parameters = {pytorch_total_params}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=config.epoch)        


#device = config.device

#logging.info(f'Training on {device}')
#net.to(device)

for epoch in range(config.epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # labels = labels.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # logging.info(inputs.shape, labels.shape, outputs.shape)
        # logging.info(labels, outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # logging.info statistics
        running_loss += loss.item()
        # correct += (outputs == labels).float().sum()
        if i % 2000 == 1999:    # logging.info every 2000 mini-batches
            logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    # accuracy = 100 * correct / len(trainset)
    # logging.info("Accuracy = {}".format(accuracy))
logging.info(f'Finished Training')
logging.info(f'Optimal Learning rate is {scheduler.get_last_lr()}')



correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

logging.info(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

torch.save(net.state_dict(), config.PATH["MODEL_SAVE"])

