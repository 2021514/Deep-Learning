import torch
import torchaudio
import torchvision
import torch.nn as nn
from Pipeline import *
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchaudio.datasets import SPEECHCOMMANDS
import math
from sklearn.preprocessing import LabelEncoder
image_dataset_downloader = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,  
    transform=transforms.ToTensor(),
    download=True
)
audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(
    root='./data', 
    url='speech_commands_v0.02',  
    download=True
)

class ImageDataset(Dataset):
    def __init__(self, split: str = "train",train_percentage=0.8, transform=None) -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        full_dataset = image_dataset_downloader
        total_size = len(full_dataset)
        train_size = int(total_size*train_percentage)
        val_size = total_size-train_size
        torch.manual_seed(0)
        train_dataset, val_dataset = random_split(full_dataset,[train_size,val_size])
        image_dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

        test_dataset = image_dataset_test
        
        if split == "train":
            self.dataset = train_dataset
        elif split == "val":
            self.dataset = val_dataset
        elif split == "test":
            self.dataset = test_dataset
        
        self.transform = transform
        self.dataset_size = len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

class AudioDataset(Dataset):
    def __init__(self, root: str = "./data", split: str = "train", train_percentage=0.8, test_percentage=0.1, transform=None):
        self.dataset = SPEECHCOMMANDS(root=root, url='speech_commands_v0.02', download=True)
        full_dataset = self.dataset
        self.datasplit = split
        self.train_percentage = train_percentage
        self.transform = transform
        self.labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        total_size = len(full_dataset)
        train_size = int(total_size * train_percentage)
        temp_size = total_size - train_size  # TEMP is the dataset to be split into TEST and VALIDATION
        test_size = int(temp_size * (test_percentage / (1 - train_percentage)))  # Adjust test size based on remaining dataset
        val_size = temp_size - test_size

        torch.manual_seed(0)
        
        train_dataset, temp_dataset = random_split(full_dataset, [train_size, temp_size])
        
        test_dataset, val_dataset = random_split(temp_dataset, [test_size, val_size])
        
        if split == "train":
            self.dataset = train_dataset
        elif split == "val":
            self.dataset = val_dataset
        elif split == "test":
            self.dataset = test_dataset
        
        self.transform = transform
        self.dataset_size = len(self.dataset)
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        if self.datasplit == "test":
            audio, _, label, *_ = self.dataset[idx]
        else:
            audio, _, label, *_ = self.dataset[idx]
        
        # Encode labels
        label = self.label_encoder.transform([label])[0]
        
        # Handling data points with less features (Padding)
        if audio.shape[1] < 16000:
            padding = torch.zeros(1, 16000 - audio.shape[1])
            audio = torch.cat((audio, padding), dim=1)
        
        if self.transform:
            audio = self.transform(audio)
        
        if self.datasplit == "test":
            return audio, label
        else:
            return audio, label

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_1d=False):
        super(ResNetBlock, self).__init__()
        if use_1d:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # print(out.shape)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # print(out.shape)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
class Resnet_Q1(nn.Module):
    def __init__(self):
        super(Resnet_Q1, self).__init__()
        cout=1
        self.conv2d = nn.Conv2d(3, cout, kernel_size=3, padding=1)
        self.conv1d = nn.Conv1d(1, cout, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(cout)
        self.bn1d = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU()
        self.residual_blocks1d = nn.Sequential(*[ResNetBlock(cout, cout, True) for _ in range(18)])
        self.residual_blocks2d = nn.Sequential(*[ResNetBlock(cout, cout, False) for _ in range(18)])
        self.fc1 = nn.Linear(cout*16000,35)
        self.fc2 = nn.Linear(cout*32*32,10)

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1: # for audio
            out = self.conv1d(x)
            # print(out.shape)
            out = self.bn1d(out)
            out = self.relu(out)
            out = self.residual_blocks1d(out)
            # print(out.shape)
            out = out.view(out.size(0), -1)
            # print(out.shape)
            out = self.fc1(out)
        else: #for image
            out = self.conv2d(x)
            out = self.bn2d(out)
            out = self.relu(out)
            out = self.residual_blocks2d(out)
            out = out.view(out.size(0), -1)
            out = self.fc2(out)
        return out

        
class VGG_Q2(nn.Module):
    def __init__(self):
        super(VGG_Q2, self).__init__()
        initial_channels = 8
        initial_kernel_size = 3
        channels = [int(math.ceil(initial_channels * (0.65 ** i))) for i in range(5)]
        kernels = [int(math.ceil(initial_kernel_size * (1.25 ** i))) for i in range(5)]
        self.audio_architecture = nn.Sequential(
            self.Convolution_block_audio(1, channels[0], kernels[0], 2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self.Convolution_block_audio(channels[0], channels[1], kernels[1], 2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self.Convolution_block_audio(channels[1], channels[2], kernels[2], 3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self.Convolution_block_audio(channels[2], channels[3], kernels[3], 3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self.Convolution_block_audio(channels[3], channels[4], kernels[4], 3),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.image_architecture = nn.Sequential(
            self.Convolution_block_image(3, channels[0], kernels[0], 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.Convolution_block_image(channels[0], channels[1], kernels[1], 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.Convolution_block_image(channels[1], channels[2], kernels[2], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.Convolution_block_image(channels[2], channels[3], kernels[3], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.Convolution_block_image(channels[3], channels[4], kernels[4], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.audio_dense_layer = nn.Sequential(
                        nn.Linear(1004, 256),
                        nn.ReLU(True),
                        nn.Linear(256, 128),
                        nn.ReLU(True),
                        nn.Linear(128, 35),  # Assuming 35 classes for audio
                    )#.to(x.device)
        self.image_dense_layer  = nn.Sequential(
                        nn.Linear(18, 64),
                        nn.ReLU(True),
                        nn.Linear(64, 64),
                        nn.ReLU(True),
                        nn.Linear(64, 10),  # Assuming 10 classes for images
                    )#.to(x.device)


    def Convolution_block_audio(self, in_channels, out_channels, kernel_size, convs):
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(1, convs):
            layers.extend([
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)])
        return nn.Sequential(*layers)
    
    def Convolution_block_image(self, in_channels, out_channels, kernel_size, convs):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        for _ in range(1, convs):
            layers.extend([nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)])
        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            
            x = self.audio_architecture(x).to(x.device)
            x = torch.flatten(x, 1)
            x = self.audio_dense_layer(x).to(x.device)
        else:  # Image data
            x = self.image_architecture(x).to(x.device)
            x = torch.flatten(x, 1)
            x = self.image_dense_layer(x).to(x.device)

        return x
class Block_inception(nn.Module):
    def __init__(self, in_channels, out_channels, is1d = False):
        super(Block_inception, self).__init__()
        self.is1d = is1d

        # Branches for Images
        self.branch1_2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2_2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5,padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3_2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5,padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch4_2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Branches for audio
        self.branch1_1d = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2_1d = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5,padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3_1d = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5,padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch4_1d = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        if self.is1d: # Audio
            branch1 = self.branch1_1d(x)
            branch2 = self.branch2_1d(x)
            branch3 = self.branch3_1d(x)
            branch4 = self.branch4_1d(x)

        else: #Image
            branch1 = self.branch1_2d(x)
            branch2 = self.branch2_2d(x)
            branch3 = self.branch3_2d(x)
            branch4 = self.branch4_2d(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)


class Inception_Q3(nn.Module):
    def __init__(self, in_channels=3):
        super(Inception_Q3, self).__init__()

        self.inception_blocks_2d = nn.Sequential(
            Block_inception(in_channels, 8),
            Block_inception(27, 12),
            Block_inception(63, 8),
            Block_inception(87, 87)
        )
        self.fc_2d = nn.Linear(356352, 10)

        self.inception_blocks_1d = nn.Sequential(
            Block_inception(1, 2,True),
            Block_inception(7, 2,True),
            Block_inception(13, 2,True),
            Block_inception(19, 2,True)
        )
        self.fc_1d = nn.Linear(400000, 35)

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = self.inception_blocks_1d(x)
            x = x.view(x.size(0),-1)
            x = self.fc_1d(x)
        else:
            x = self.inception_blocks_2d(x)
            x = x.view(x.size(0),-1)
            x = self.fc_2d(x)

        return x

        
class CustomNetwork_Q4(nn.Module):
    def __init__(self, in_channels=3):
        super(CustomNetwork_Q4, self).__init__()

        self.residual_blocks11 = ResNetBlock(in_channels,3,False)
        self.residual_blocks12 = ResNetBlock(3,3,False)
        # Inception Block × 2
        self.inception_blocks21 = Block_inception(3,5,False)
        self.inception_blocks22 = Block_inception(18,5,False)
        # Residual Block × 1
        self.residual_blocks31 = ResNetBlock(33,33, False)

        self.inception_blocks41 = Block_inception(33,15,False)
        
        self.residual_blocks51 = ResNetBlock(78,78, False)
        self.inception_blocks61 = Block_inception(78,20,False)

        self.residual_blocks71 = ResNetBlock(138,138, False)
        self.inception_blocks81 = Block_inception(138,32,False)

        self.fc_layer_img = nn.Linear(239616,10)

        self.audio_channel = 1
        self.residual_blocks11_aud = ResNetBlock(self.audio_channel,1,True)
        self.residual_blocks12_aud = ResNetBlock(1,1,True)
        # Inception Block × 2
        self.inception_blocks21_aud = Block_inception(1,1,True)
        self.inception_blocks22_aud = Block_inception(4,1,True)
        # Residual Block × 1
        self.residual_blocks31_aud = ResNetBlock(7,7, True)

        self.inception_blocks41_aud = Block_inception(7,1,True)
        
        self.residual_blocks51_aud = ResNetBlock(10,10, True)
        self.inception_blocks61_aud = Block_inception(10,1,True)

        self.residual_blocks71_aud = ResNetBlock(13,13, True)
        self.inception_blocks81_aud = Block_inception(13,1,True)

        self.fc_layer_aud = nn.Linear(256000,35)

        # Classification Network
        self.classifier = nn.Sequential(
            nn.Linear(239616,10)

        )

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = self.residual_blocks11_aud(x)
            x = self.residual_blocks12_aud(x)
            x = self.inception_blocks21_aud(x)
            x = self.inception_blocks22_aud(x)
            x = self.residual_blocks31_aud(x)
            x = self.inception_blocks41_aud(x)
            x = self.residual_blocks51_aud(x)
            x = self.inception_blocks61_aud(x)
            x = self.residual_blocks71_aud(x)
            x = self.inception_blocks81_aud(x)
            x = x.view(x.size(0),-1)
            x = self.fc_layer_aud(x)
        else:
            x = self.residual_blocks11(x)
            x = self.residual_blocks12(x)
            x = self.inception_blocks21(x)
            x = self.inception_blocks22(x)
            x = self.residual_blocks31(x)
            x = self.inception_blocks41(x)
            x = self.residual_blocks51(x)
            x = self.inception_blocks61(x)
            x = self.residual_blocks71(x)
            x = self.inception_blocks81(x)
            x = x.view(x.size(0),-1)
            x = self.fc_layer_img(x)

        return x

def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    
    device = torch.device("cuda:0" if gpu == "T" else "cpu")
    network = network.to(device)

    for epoch in range(EPOCH):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            # inputs, labels = inputs.to(device), labels.to(device)
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # print('loss=',loss.item())
        
        accuracy = 100. * correct / total
        
        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch+1,
            epoch_loss,
            accuracy
        ))
    torch.save({
        'model_state_dict':network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, "checkpoint.pth")

def validator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None,
              t_epoch=1):
    
    device = torch.device("cuda:0" if gpu == "T" else "cpu")
    network = network.to(device)
    
    checkpoint = torch.load("checkpoint.pth")
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(t_epoch):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = network(inputs)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total

        print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch+1,
            epoch_loss,
            accuracy
        ))
    torch.save(network.state_dict(), "checkpoint.pth")


def evaluator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None,
              t_epoch=1):
    device = torch.device("cuda:0" if gpu == "T" else "cpu")
    network = network.to(device)

    checkpoint = torch.load("checkpoint.pth")
    network.load_state_dict(checkpoint)

    if criterion==None:
        criterion = nn.CrossEntropyLoss()
    for epoch in range(t_epoch):
        epoch_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = network(inputs)
                loss = criterion(outputs, labels)

                epoch_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total

        print("[Loss: {}, Accuracy: {}]".format(
            epoch_loss,
            accuracy
        ))

