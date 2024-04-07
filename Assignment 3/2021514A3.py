import os,torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torchvision.utils import save_image as saveimage
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from skimage import io, transform as sk_transform
from torchvision.io import read_image, ImageReadMode
from . import *
checker = 5

class AlteredMNIST:
    def __init__(self):
        self.aug_path = "Data/aug"
        self.clean_path = "Data/clean"
        self.aug_file_list = [os.path.join(self.aug_path, filename) for filename in os.listdir(self.aug_path)]
        self.clean_file_list = [os.path.join(self.clean_path, filename) for filename in os.listdir(self.clean_path)]
        self.Aug=[]
        self.Clean=[]
        self.make_pair()

    def imgpath_to_tensor(self, path):
        img = read_image(path, ImageReadMode.GRAY)
        image_tensor = img.float()
        image_tensor = image_tensor.mean(dim=0).unsqueeze(0)
        image_tensor = image_tensor.cpu().numpy()
        image_tensor = image_tensor[0]
        image_tensor = image_tensor.reshape(-1)
        return image_tensor
    
    def make_pair(self):
        aug_img_class ={}
        clean_img_class ={}
        for i in range(10):
            aug_img_class[i]=[]
            clean_img_class[i]=[]
        for aug_path in self.aug_file_list:
            aug_img_class[int(aug_path.split("/")[-1][-5])].append(aug_path) 
        for clean_path in self.clean_file_list:
            clean_img_class[int(clean_path.split("/")[-1][-5])].append(clean_path)
        for i in range(10):
            gmm = KMeans(n_clusters=500, n_init=20)
            aug_images_features = [self.imgpath_to_tensor(img) for img in aug_img_class[i]]
            clean_images_features = [self.imgpath_to_tensor(img) for img in clean_img_class[i]]
            pca = PCA(n_components=15)

            clean_images_features = pca.fit_transform(clean_images_features)
            aug_images_features = pca.transform(aug_images_features)

            clean_labels = gmm.fit_predict(clean_images_features)
            aug_labels = gmm.predict(aug_images_features)

            clean_dict = {}
            for j, clean_label in enumerate(clean_labels):
                if clean_label not in clean_dict:
                    clean_dict[clean_label] = [clean_img_class[i][j]]
                else:
                    clean_dict[clean_label].append(clean_img_class[i][j])

            for j, aug_label in enumerate(aug_labels):
                for k in clean_dict[aug_label]:
                    self.Aug.append(aug_img_class[i][j])
                    self.Clean.append(k)

    def __len__(self):
        return len(self.Aug)

    def __getitem__(self, idx):
        aug = read_image(self.Aug[idx])
        clean = read_image(self.Clean[idx])
        clean_path = self.Clean[idx]
        clean_label = int(clean_path.split("/")[-1][-5])
        aug = aug.float()
        aug = aug.mean(dim=0).unsqueeze(0)
        clean = clean.float()
        clean = clean.mean(dim=0).unsqueeze(0)
        aug = aug / 255.0
        clean = clean / 255.0
        return aug, clean, clean_label

def path_to_img(path):
    img = read_image(path)
    img = img.float()
    img = img.mean(dim=0).unsqueeze(0)
    img = img / 255.0
    return img

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, decode=False):
        super(ResNetBlock, self).__init__()
        self.adjust_channels = (in_channels != out_channels)
        if decode == True:
            self.conv1 = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
            if self.adjust_channels:
                self.identity_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
            if self.adjust_channels:
                self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.adjust_channels:
            identity = self.identity_conv(identity)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.block1 = ResNetBlock(1, 2, 4)
        self.block2 = ResNetBlock(4, 8, 16)
        self.nn = nn.Sequential(nn.Linear(16 * 28 * 28, 128), nn.BatchNorm1d(128), nn.ReLU())

        self.v_mean_layer = nn.Linear(128, 128)
        self.v_var_layer = nn.Linear(128, 128)

        self.num_classes = 10
        self.cv_mean_layer = nn.Linear(128, 128)
        self.cv_var_layer = nn.Linear(128, 128)
        self.label_projector = nn.Sequential(
                nn.Linear(self.num_classes, 128),
                nn.ReLU(),
            )
    def forward(self, x,y=None):
        if checker == 1:
            out = self.block1(x)
            out = self.block2(out)
            out = out.view(out.size(0), -1)
            out = self.nn(out)
            return out
        if checker == 2:
            out = self.block1(x)
            out = self.block2(out)
            out = out.view(out.size(0), -1)
            out = self.nn(out)
            return self.v_mean_layer(out), self.v_var_layer(out)
        if checker==3:
            out = self.block1(x)
            out = self.block2(out)
            out = out.view(out.size(0), -1)
            out = self.nn(out)
            v_mean = self.cv_mean_layer(out)
            v_log_var = self.cv_var_layer(out)            
            return v_mean, v_log_var


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.e_nn = nn.Sequential(nn.Linear(128, 16*28*28), nn.BatchNorm1d(16*28*28), nn.ReLU())
        self.block2 = ResNetBlock(16, 8, 4, True)
        self.block1 = ResNetBlock(4, 2, 1, True)
        self.num_classes = 10
        self.projection = nn.Sequential(nn.Linear(10,128), nn.ReLU())
        self.v_nn = nn.Sequential(nn.Linear(128, 16*28*28), nn.BatchNorm1d(16*28*28), nn.ReLU())
    def forward(self, x,y=None):
        if checker == 1:
            out = self.e_nn(x)
            out = out.view(out.size(0), 16, 28, 28)
            out = self.block2(out)
            out = self.block1(out)
            return out
        if checker == 2:
            out = self.v_nn(x)
            out = out.view(out.size(0), 16, 28, 28)
            out = self.block2(out)
            out = self.block1(out)
            return out
        if checker==3:
            y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
            y_one_hot = self.projection(y_one_hot)
            x+=y_one_hot
            x = self.v_nn(x)
            x = x.view(x.size(0), 16, 28, 28)
            x = self.block2(x)
            x = self.block1(x)
            return x

class AELossFn(nn.Module):
    def __init__(self):
        super(AELossFn, self).__init__()

    def forward(self, target, output):
        return F.mse_loss(output, target, reduction='mean')

class VAELossFn(nn.Module):
    def __init__(self):
        super(VAELossFn, self).__init__()

    def forward(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD

def ParameterSelector(E, D):
    return list(E.parameters()) + list(D.parameters())


class AETrainer:
    def __init__(self, dataset, encoder, decoder, loss_fn, optimizer, gpu):
        super(AETrainer, self).__init__()
        global checker
        checker = 1
        self.dataset = dataset
        self.encoder = encoder
        self.decoder = decoder
        loss_fn = AELossFn()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device("cuda" if (torch.cuda.is_available() and gpu == 'T') else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.loss_fn.to(self.device)
        self.train(EPOCH)

    def compute_similarity(self, output, target):
        output = output.squeeze().cpu().detach().numpy()
        target = target.squeeze().cpu().detach().numpy()
        ssim_scores = [ssim(output[i], target[i], data_range=(target[i].max() - target[i].min())) for i in range(output.shape[0])]
        return np.mean(ssim_scores)
    
    def train(self, num_epochs):
        global checker
        checker = 1
        for epoch in range(1, num_epochs + 1):
            self.encoder.train()
            self.decoder.train()
            total_loss = 0.0
            total_similarity = 0.0
            for minibatch, data in enumerate(self.dataset):
                aug_image, clean_image,label = data
                aug_image = aug_image.to(self.device)
                clean_image = clean_image.to(self.device)
                self.optimizer.zero_grad()
                encoded = self.encoder(aug_image)
                decoded = self.decoder(encoded)
                loss = self.loss_fn(decoded, clean_image)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                similarity = self.compute_similarity(decoded, clean_image)
                total_similarity += similarity
                if minibatch % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
            
            avg_loss = total_loss / (len(self.dataset)) 
            avg_similarity = total_similarity / (len(self.dataset))
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, avg_loss, avg_similarity))

            if epoch % 10 == 0:
                self.visualize_tsne(epoch)
            if epoch == EPOCH:
                checkpoints = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                }
                torch.save(checkpoints, 'AE_checkpoint.pth') 
    def visualize_tsne(self, epoch):
        embeddings = []
        with torch.no_grad():
            for aug_images, _ in self.dataset:
                aug_images = aug_images.to(self.device)
                encoded = self.encoder(aug_images)
                embeddings.append(encoded.view(encoded.size(0), -1).cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        
        tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, init='pca')
        tsne_features = tsne.fit_transform(embeddings)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_features[:, 0], tsne_features[:, 1], tsne_features[:, 2], alpha=0.5)
        plt.savefig(f'AE_epoch_{epoch}_TSNE.png')
        ax.set_xlabel('TSNE Component 1')
        ax.set_ylabel('TSNE Component 2')
        ax.set_zlabel('TSNE Component 3')
        ax.set_title(f'3D TSNE Plot for {epoch}')
        plt.show()

class VAETrainer:
    def __init__(self, dataset, encoder, decoder, loss_fn, optimizer, gpu):
        super(VAETrainer, self).__init__()
        global checker
        checker = 2
        self.dataset = dataset
        self.encoder = encoder
        self.decoder = decoder
        loss_fn = VAELossFn()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device("cuda" if (torch.cuda.is_available() and gpu == 'T') else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.loss_fn.to(self.device)
        self.train(EPOCH)

    def compute_similarity(self, output, target):
        output = output.squeeze().cpu().detach().numpy()
        target = target.squeeze().cpu().detach().numpy()
        ssim_scores = [ssim(output[i], target[i], data_range=(target[i].max() - target[i].min())) for i in range(output.shape[0])]
        return np.mean(ssim_scores)

    def reparameterize(self, mean, variance):
        return mean + torch.randn_like(variance) * variance
    
    def train(self, num_epochs):
        global checker
        checker = 2
        for epoch in range(1, num_epochs + 1):
            self.encoder.train()
            self.decoder.train()
            total_loss = 0.0
            total_similarity = 0.0
            for minibatch, data in enumerate(self.dataset):
                aug_image, clean_image,label = data
                aug_image = aug_image.to(self.device)
                clean_image = clean_image.to(self.device)
                self.optimizer.zero_grad()
                mean, variance = self.encoder(aug_image)
                output = self.decoder(self.reparameterize(mean, variance))
                loss = self.loss_fn(clean_image, output, mean, variance)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                similarity = self.compute_similarity(output, clean_image)
                total_similarity += similarity

                if minibatch % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch, loss, similarity))

            avg_loss = total_loss / (len(self.dataset)) 
            avg_similarity = total_similarity / (len(self.dataset))
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, avg_loss, avg_similarity))

            if epoch % 10 == 0:
                self.visualize_tsne(epoch)
            if epoch == EPOCH:
                checkpoints = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                }
                torch.save(checkpoints, 'VAE_checkpoint.pth')
    def visualize_tsne(self, epoch):
        embeddings = []
        with torch.no_grad():
            for aug_images, _ in self.dataset:
                aug_images = aug_images.to(self.device)
                mean, variance = self.encoder(aug_images)
                output = self.reparameterize(mean, variance)
                embeddings.append(output.view(output.size(0), -1).cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        
        tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, init='pca')
        tsne_features = tsne.fit_transform(embeddings)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_features[:, 0], tsne_features[:, 1], tsne_features[:, 2], alpha=0.5)
        plt.savefig(f'VAE_epoch_{epoch}_TSNE.png')
        ax.set_xlabel('TSNE Component 1')
        ax.set_ylabel('TSNE Component 2')
        ax.set_zlabel('TSNE Component 3')
        ax.set_title(f'3D TSNE Plot for {epoch}')
        plt.show()

class AE_TRAINED:
    def __init__(self,gpu=False):
        global checker
        checker = 1
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.device = torch.device("cuda" if (torch.cuda.is_available() and gpu == True) else "cpu")
        checkpoint = torch.load('AE_checkpoint.pth',map_location=torch.device('cpu'))

        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()

        # print("Loaded final encoder and decoder models.")

    def from_path(self, sample, original, type):
        global checker
        checker = 1
        if isinstance(sample, str):
            sample = path_to_img(sample)
            original = path_to_img(original)
        sample = sample.unsqueeze(0)
        original = original.unsqueeze(0)
        if sample.max()>1:
            sample = sample/255.0
            original = original/255.0
        self.encoder.eval()
        self.decoder.eval()
        encoded_img = self.encoder(sample)
        recon_img = self.decoder(encoded_img)
        recon_img = recon_img.squeeze(0)
        original = original.squeeze(0)

        if type == "SSIM":
            return structure_similarity_index(recon_img, original)
        elif type == "PSNR":
            return peak_signal_to_noise_ratio(recon_img, original)

class VAE_TRAINED:
    def __init__(self,gpu=False):
        global checker
        checker = 2
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.device = torch.device("cuda" if (torch.cuda.is_available() and gpu == True) else "cpu")
        checkpoint = torch.load('VAE_checkpoint.pth',map_location=torch.device('cpu'))
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()

        # print("Loaded final encoder and decoder models.")
    def reparameterize(self, mean, variance):
        return mean + torch.randn_like(variance) * variance
    def from_path(self,sample, original, type):
        global checker
        checker = 2
        if isinstance(sample, str):
            sample = path_to_img(sample)
            original = path_to_img(original)
        sample = sample.unsqueeze(0)
        original = original.unsqueeze(0)
        if sample.max()>1:
            sample = sample/255.0
            original = original/255.0
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            mean, variance = self.encoder(sample)
            recon_img = self.decoder(self.reparameterize(mean, variance))
            recon_img = recon_img.squeeze(0)
            original = original.squeeze(0)
            if type == "SSIM":
                return structure_similarity_index(recon_img, original)
            elif type == "PSNR":
                return peak_signal_to_noise_ratio(recon_img, original)

class CVAELossFn(nn.Module):
    def __init__(self):
        super(CVAELossFn, self).__init__()

    def forward(self, x, x_hat, mean, log_var):
        BCE = F.mse_loss(x_hat, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return BCE + KLD

class CVAE_Trainer:
    def __init__(self, dataset, encoder, decoder, loss_fn, optimizer):
        global checker
        checker = 3
        super(CVAE_Trainer, self).__init__()
        self.dataset = dataset
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.loss_fn.to(self.device)
        self.num_classes=10
        self.train(EPOCH)
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def condition_on_label(self, z, y):
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        projected_label = self.encoder.label_projector(y_one_hot)
        return z + projected_label
    
    def compute_similarity(self, output, target):
        output = output.squeeze().cpu().detach().numpy()
        target = target.squeeze().cpu().detach().numpy()
        ssim_scores = [ssim(output[i], target[i], data_range=(target[i].max() - target[i].min())) for i in range(output.shape[0])]
        return np.mean(ssim_scores)

    def train(self, num_epochs):
        global checker
        checker = 3
        for epoch in range(1, num_epochs + 1):
            self.encoder.train()
            self.decoder.train()
            total_loss = 0.0
            total_similarity = 0.0
            for minibatch, data in enumerate(self.dataset):
                clean_image, clean_image, label = data
                clean_image, clean_image, label = clean_image.to(self.device), clean_image.to(self.device), label.to(self.device)
                mean, log_var = self.encoder(clean_image, label)
                z = self.reparameterize(mean, log_var)
                output = self.decoder(z,label)
                loss = self.loss_fn(clean_image, output, mean, log_var)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                similarity = self.compute_similarity(output, clean_image)
                total_similarity += similarity

                if minibatch % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch, loss, similarity))

            avg_loss = total_loss / (len(self.dataset)) 
            avg_similarity = total_similarity / (len(self.dataset))
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, avg_loss, avg_similarity))

            if epoch%10==0:
                self.visualize_tsne(self)
            if epoch == EPOCH:
                checkpoints = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                }
                torch.save(checkpoints, 'CVAE_checkpoint.pth')
    def visualize_tsne(self, epoch):
        embeddings = []
        with torch.no_grad():
            for aug_images, _,_ in self.dataset:
                aug_images = aug_images.to(self.device)
                mean, variance = self.encoder(aug_images)
                output = self.reparameterize(mean, variance)
                embeddings.append(output.view(output.size(0), -1).cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        
        tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, init='pca')
        tsne_features = tsne.fit_transform(embeddings)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_features[:, 0], tsne_features[:, 1], tsne_features[:, 2], alpha=0.5)
        plt.savefig(f'VAE_epoch_{epoch}_TSNE.png')
        ax.set_xlabel('TSNE Component 1')
        ax.set_ylabel('TSNE Component 2')
        ax.set_zlabel('TSNE Component 3')
        ax.set_title(f'3D TSNE Plot for {epoch}')
        plt.show()
class CVAE_Generator:
    def __init__(self):

        self.decoder = Decoder()
        self.encoder = Encoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder.to(self.device)
        self.encoder.to(self.device)

        checkpoint = torch.load('CVAE_checkpoint.pth',map_location=torch.device('cpu'))
        self.decoder.load_state_dict(checkpoint['decoder'])

        global checker
        checker = 3
        self.decoder.eval()
        self.image_count = 1
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def condition_on_label(self, z, y):
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        projected_label = self.encoder.label_projector(y_one_hot)
        return z + projected_label
    def save_image(self, digit, save_path):

        z = torch.distributions.Normal(0, 1).sample((1, 128)).to(self.device)

        label = torch.tensor([digit]).to(self.device)
        output = self.decoder(z, label)
        output = output[0]
        img_path = os.path.join(save_path, f"img_{self.image_count}_{digit}.png")
        self.image_count += 1
        saveimage(output, img_path, format='png')

def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    img1 = img1.squeeze().cpu().detach().numpy()
    img2 = img2.squeeze().cpu().detach().numpy()
    ssim_scores = [ssim(img1[i], img2[i], data_range=(img2[i].max()+1e-9 - img2[i].min())) for i in range(img1.shape[0])]
    return np.mean(ssim_scores)
