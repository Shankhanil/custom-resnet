import torch

batch_size = 4
epoch = 30

earlystop_patience = 15
# PATH['MODEL_SAVE'] = './custom-resnet-3.pth'

PATH = {
    "MODEL_SAVE": './custom-resnet-3.pth',
    "LOGS": "./logs.txt",
    # "MODEL_LOGS": ""
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
