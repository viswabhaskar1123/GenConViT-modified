import torch
import torch.nn as nn
from timm import create_model
from torchvision import transforms

class GenConViTV2(nn.Module):
    def __init__(self, config, pretrained=True):
        super(GenConViTV2, self).__init__()
        self.latent_dims = config['model']['latent_dims']

        # Backbone - ConvNeXtV2 with DeiT-3 for better accuracy
        self.convnext_backbone = create_model('convnextv2_large', pretrained=pretrained)
        self.deit_backbone = create_model('deit3_large_patch16_224', pretrained=pretrained)

        # VAE with Self-Attention
        self.encoder = Encoder(self.latent_dims)
        self.decoder = Decoder(self.latent_dims)

        # Fusion layers
        self.fc1 = nn.Linear(2048 + 1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, config['num_classes'])

        self.relu = nn.ReLU()
        self.resize = transforms.Resize((224, 224), antialias=True)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # Improved backbone features
        x1 = self.convnext_backbone(x)
        x2 = self.deit_backbone(x_hat)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

        return x, self.resize(x_hat)


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

        self.mu = nn.Linear(512 * 7 * 7, latent_dims)
        self.var = nn.Linear(512 * 7 * 7, latent_dims)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.mu(x)
        var = self.var(x)

        return mu, var


class Decoder(nn.Module):
    def __init__(self, latent_dims=4):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dims, 512 * 7 * 7)
        
        self.features = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 7, 7)
        x = self.features(x)
        return x
