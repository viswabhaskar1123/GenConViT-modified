import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from timm import create_model

class GenConViTV2(nn.Module):
    def __init__(self, config, pretrained=True):
        super(GenConViTV2, self).__init__()
        self.latent_dims = config['model']['latent_dims']

        # Backbone models
        self.convnext_backbone = create_model('convnextv2_large', pretrained=pretrained)
        self.deit_backbone = create_model('deit3_large_patch16_224', pretrained=pretrained)

        # VAE with Self-Attention
        self.encoder = Encoder(self.latent_dims)
        self.decoder = Decoder(self.latent_dims)

        # Fusion layers with reduced dimensions
        self.fc1 = nn.Linear(2048 + 1024, 512)
        self.fc2 = nn.Linear(512, 256)

        # Reduced dimensions to avoid large tensor allocation
        latent_features = min(1024, self.latent_dims * 16 * 16)
        self.fc3 = nn.Linear(256, latent_features)
        self.fc_out = nn.Linear(latent_features, config['num_classes'])

        self.relu = nn.ReLU()
        self.resize = transforms.Resize((224, 224), antialias=True)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # Extract features from backbones
        x1 = self.convnext_backbone(x)
        x2 = self.deit_backbone(x_hat)

        # Concatenate and pass through FC layers
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.fc_out(self.relu(x))

        # return x, self.resize(x_hat)
        return x, x_hat


class Encoder(nn.Module):
    def __init__(self, latent_dims=4):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Use smaller dimensions to prevent OOM
        # In Encoder class
        # self.mu = nn.Linear(512 * 7 * 7, latent_dims)
        # self.var = nn.Linear(512 * 7 * 7, latent_dims)?
        self.mu = nn.Linear(512 * 14 * 14, latent_dims)
        self.var = nn.Linear(512 * 14 * 14, latent_dims)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.mu(x)
        var = self.var(x)

        return mu, var


class Decoder(nn.Module):
    def __init__(self, latent_dims=4):
        super(Decoder, self).__init__()

        # Reduced latent features to avoid large tensors
        latent_features = min(1024, latent_dims * 16 * 16)
        self.fc = nn.Linear(latent_dims, latent_features)

        # self.features = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.Tanh()
        # )
        # In Decoder's features
        self.features = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Additional layer to reach 224x224
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 7, 7)
        x = self.features(x)
        return x
