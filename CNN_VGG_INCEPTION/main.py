# Replace changerollno with your rollnumber as mentioned in Assignment Guidelines
import argparse
from Pipeline import *
from Pipeline.CNN_Architectures import *

P = argparse.ArgumentParser()
P.add_argument("gpu", type=str)
A = P.parse_args()
    

if __name__ == "__main__":
    
    imageDataset = [
        ImageDataset(split="train"),
        ImageDataset(split="val"),
        ImageDataset(split="test")
        
    ]
    
    audioDataset = [
        AudioDataset(split="train"),
        AudioDataset(split="val"),
        AudioDataset(split="test")
    ]
    
    Architectures = [
        Resnet_Q1(),
        VGG_Q2(),
        Inception_Q3(),
        CustomNetwork_Q4()
    ]
    
    
    for network in Architectures:
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            params=network.parameters(),
            lr=LEARNING_RATE
        )
        
        for dataset in imageDataset + audioDataset:
            if dataset.datasplit == "train":
                print(
                    "Training {} Architecture on {} split of {}".format(
                        network.__class__.__name__,
                        dataset.datasplit,
                        dataset.__class__.__name__
                    )
                )
                network.train()
                train_dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True
                )
                
                trainer(
                    gpu=A.gpu,
                    dataloader=train_dataloader,
                    network=network,
                    criterion=criterion,
                    optimizer=optimizer
                )
            
            elif dataset.datasplit == "val":
                print(
                    "Validating {} Architecture on {} split of {}".format(
                        network.__class__.__name__,
                        dataset.datasplit,
                        dataset.__class__.__name__
                    )
                )
                network.train()
                val_dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True
                )
                
                validator(
                    gpu=A.gpu,
                    dataloader=val_dataloader,
                    network=network,
                    criterion=criterion,
                    optimizer=optimizer
                )
                
            elif dataset.datasplit == "test":
                print(
                    "Testing {} Architecture on {} split of {}".format(
                        network.__class__.__name__,
                        dataset.datasplit,
                        dataset.__class__.__name__
                    )
                )
                network.eval()
                test_dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True
                )
                evaluator(
                    gpu=A.gpu,
                    dataloader=test_dataloader,
                    network=network,
                    criterion=None,
                    optimizer=None)
