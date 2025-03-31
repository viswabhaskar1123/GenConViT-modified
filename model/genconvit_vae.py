# # original
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from timm import create_model
# from model.config import load_config
# from .model_embedder import HybridEmbed

# config = load_config()

# class Encoder(nn.Module):

#     def __init__(self, latent_dims=4):
#         super(Encoder, self).__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(num_features=16),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(num_features=32),
#             nn.LeakyReLU(),
            
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(num_features=64),
#             nn.LeakyReLU(),

#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(num_features=128),
#             nn.LeakyReLU()
#         )

#         self.latent_dims = latent_dims
#         self.fc1 = nn.Linear(128*14*14, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.mu = nn.Linear(128*14*14, self.latent_dims)
#         self.var = nn.Linear(128*14*14, self.latent_dims)

#         self.kl = 0
#         self.kl_weight = 0.5#0.00025
#         self.relu = nn.LeakyReLU()

#     def reparameterize(self, x):
#         # https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vanilla_vae.py
#         std = torch.exp(0.5*self.mu(x))
#         eps = torch.randn_like(std)
#         z = eps * std + self.mu(x)

#         return z, std

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, start_dim=1)

#         mu =  self.mu(x)
#         var = self.var(x)
#         z,_ = self.reparameterize(x)
#         self.kl = self.kl_weight*torch.mean(-0.5*torch.sum(1+var - mu**2 - var.exp(), dim=1), dim=0) 
        
#         return z

# class Decoder(nn.Module):
  
#     def __init__(self, latent_dims=4):
#         super(Decoder, self).__init__()

#         self.features = nn.Sequential(
#             nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
#             nn.LeakyReLU(),

#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
#             nn.LeakyReLU(),

#             nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
#             nn.LeakyReLU(),

#             nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
#             nn.LeakyReLU()
#         )

#         self.latent_dims = latent_dims
        
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 7, 7))

#     def forward(self, x): 
#         x = self.unflatten(x)
#         x = self.features(x)
#         return x
        
# class GenConViTVAE(nn.Module):
#     def __init__(self, config, pretrained=True):
#         super(GenConViTVAE, self).__init__()
#         self.latent_dims = config['model']['latent_dims']
#         self.encoder = Encoder(self.latent_dims)
#         self.decoder = Decoder(self.latent_dims)
#         self.embedder = create_model(config['model']['embedder'], pretrained=True)
#         self.convnext_backbone = create_model(config['model']['backbone'], pretrained=True, num_classes=1000, drop_path_rate=0, head_init_scale=1.0)
#         self.convnext_backbone.patch_embed = HybridEmbed(self.embedder, img_size=config['img_size'], embed_dim=768)
#         self.num_feature = self.convnext_backbone.head.fc.out_features * 2
 
#         self.fc = nn.Linear(self.num_feature, self.num_feature//4)
#         self.fc3 = nn.Linear(self.num_feature//2, self.num_feature//4)
#         self.fc2 = nn.Linear(self.num_feature//4, config['num_classes'])
#         self.relu = nn.ReLU()
#         self.resize = transforms.Resize((224,224), antialias=True)
#         self.resize_output = transforms.Resize((600, 600), antialias=True)

#     def forward(self, x):
#         x = self.resize(x)
#         z = self.encoder(x)
#         x_hat = self.decoder(z)

#         x1 = self.convnext_backbone(x)
#         x2 = self.convnext_backbone(x_hat)
#         x = torch.cat((x1,x2), dim=1)
#         x = self.fc2(self.relu(self.fc(self.relu(x))))
        
#         return x, self.resize(x_hat)
#modified
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from timm import create_model
# from model.config import load_config
# from .model_embedder import HybridEmbed

# config = load_config()
# import torch
# import torch.nn as nn

# class Encoder(nn.Module):
#     """Encoder with ConvNeXt V2 and Vision Transformer features"""

#     def __init__(self, latent_dims=4):
#         super(Encoder, self).__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # More filters for richer features
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(),

#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),

#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(),

#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU()
#         )

#         # Bottleneck with latent dimensions
#         self.latent_dims = latent_dims
#         self.fc1 = nn.Linear(512 * 14 * 14, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.mu = nn.Linear(512, latent_dims)
#         self.var = nn.Linear(512, latent_dims)

#         self.kl_weight = 0.5
#         self.relu = nn.LeakyReLU()

#     def reparameterize(self, mu, var):
#         """Reparameterization trick for VAE"""
#         std = torch.exp(0.5 * var)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, start_dim=1)

#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))

#         mu = self.mu(x)
#         var = self.var(x)
        
#         z = self.reparameterize(mu, var)

#         kl_div = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp(), dim=1)
#         kl_div = self.kl_weight * kl_div.mean()

#         return z, kl_div


# class Decoder(nn.Module):
#     """Decoder for reconstructing image from latent space"""

#     def __init__(self, latent_dims=4):
#         super(Decoder, self).__init__()

#         self.latent_dims = latent_dims

#         # Linear layers to reshape the latent space back into image dimensions
#         self.fc = nn.Sequential(
#             nn.Linear(latent_dims, 512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(),
#             nn.Linear(1024, 512 * 14 * 14),
#             nn.LeakyReLU()
#         )

#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 14, 14))

#         # Deconvolution layers for upsampling
#         self.features = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(),

#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),

#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(),

#             nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()  # Output scaled between -1 and 1
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         x = self.unflatten(x)
#         x = self.features(x)
#         return x

# class GenConViTVAE(nn.Module):
#     def __init__(self, config, pretrained=True):
#         super(GenConViTVAE, self).__init__()
        
#         self.latent_dims = config['model']['latent_dims']

#         # Upgraded Backbone and Embedder
#         self.encoder = Encoder(self.latent_dims)
#         self.decoder = Decoder(self.latent_dims)

#         # Use Vision Transformer (ViT)
#         self.embedder = create_model(config['model']['embedder'], pretrained=True)
        
#         # Use ConvNeXt V2 instead of ConvNeXt
#         self.convnext_backbone = create_model(
#             config['model']['backbone'], pretrained=True, num_classes=1000, drop_path_rate=0, head_init_scale=1.0
#         )

#         # Patch embedding with HybridEmbed using ViT
#         self.convnext_backbone.patch_embed = HybridEmbed(
#             self.embedder, img_size=config['img_size'], embed_dim=768
#         )

#         # Update feature extraction with the new models
#         self.num_feature = self.convnext_backbone.head.fc.out_features * 2

#         self.fc = nn.Linear(self.num_feature, self.num_feature // 4)
#         self.fc2 = nn.Linear(self.num_feature // 4, config['num_classes'])
#         self.relu = nn.ReLU()
#         self.resize = transforms.Resize((224, 224), antialias=True)

#     def forward(self, x):
#         z, kl_div = self.encoder(x) 
#         x_hat = self.decoder(z)

#         # Feature extraction with upgraded backbone and ViT embedder
#         x1 = self.convnext_backbone(x)
#         x2 = self.convnext_backbone(x_hat)

#         x = torch.cat((x1, x2), dim=1)
#         x = self.fc2(self.relu(self.fc(self.relu(x))))
        
#         return x, x_hat, kl_div
#modify-2.0
import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from model.config import load_config
from .model_embedder import HybridEmbed

config = load_config()

class Encoder(nn.Module):
    def __init__(self, latent_dims=4):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.SiLU(),  # Replaced LeakyReLU with SiLU (Swish)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.SiLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.SiLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.SiLU()
        )

        self.latent_dims = latent_dims
        self.fc1 = nn.Linear(128*14*14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mu = nn.Linear(128*14*14, self.latent_dims)
        self.var = nn.Linear(128*14*14, self.latent_dims)

        self.kl = 0
        self.kl_weight = 0.5  # 0.00025
        self.relu = nn.SiLU()  # Replaced LeakyReLU with SiLU

    def reparameterize(self, x):
        # https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vanilla_vae.py
        std = torch.exp(0.5*self.mu(x))
        eps = torch.randn_like(std)
        z = eps * std + self.mu(x)

        return z, std

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.mu(x)
        var = self.var(x)
        z, _ = self.reparameterize(x)
        self.kl = self.kl_weight*torch.mean(-0.5*torch.sum(1+var - mu**2 - var.exp(), dim=1), dim=0) 
        
        return z, mu, var  # Return mu and var for contrastive loss

class Decoder(nn.Module):
    def __init__(self, latent_dims=4):
        super(Decoder, self).__init__()

        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.SiLU(),  # Replaced LeakyReLU with SiLU

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.SiLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.SiLU(),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.SiLU()
        )

        self.latent_dims = latent_dims
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 7, 7))

    def forward(self, x): 
        x = self.unflatten(x)
        x = self.features(x)
        return x

# Added Discriminator for VAE-GAN hybrid approach
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Added multi-scale feature fusion module
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.act = nn.SiLU()
        
    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        
        x = torch.cat([x1, x3, x5], dim=1)
        x = self.fusion(x)
        x = self.act(x)
        return x

class GenConViTVAE(nn.Module):
    def __init__(self, config, pretrained=True):
        super(GenConViTVAE, self).__init__()
        self.latent_dims = config['model']['latent_dims']
        self.encoder = Encoder(self.latent_dims)
        self.decoder = Decoder(self.latent_dims)
        
        # Upgraded backbone models
        self.embedder = create_model(config['model']['embedder'], pretrained=True)
        self.convnext_backbone = create_model(config['model']['backbone'], pretrained=True, num_classes=1000)
        
        self.convnext_backbone.patch_embed = HybridEmbed(self.embedder, img_size=config['img_size'], embed_dim=768)
        self.num_feature = self.convnext_backbone.head.fc.out_features * 2
        
        # Added discriminator for VAE-GAN hybrid approach
        self.discriminator = Discriminator()
        
        # Added multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(768, 384)
        
        # Enhanced fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.num_feature, self.num_feature//2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_feature//2, self.num_feature//4),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        
        self.fc2 = nn.Linear(self.num_feature//4, config['num_classes'])
        self.relu = nn.SiLU()  # Replaced ReLU with SiLU (Swish)
        self.resize = transforms.Resize((224,224), antialias=True)
        self.resize_output = transforms.Resize((600, 600), antialias=True)
        
        # Contrastive loss weight
        self.contrastive_weight = config.get('contrastive_weight', 0.2)

    def contrastive_loss(self, z_real, z_fake):
        # Simple contrastive loss implementation
        return torch.mean((z_real - z_fake)**2)

    def forward(self, x, target=None):
        x = self.resize(x)
        z, mu, var = self.encoder(x)
        x_hat = self.decoder(z)
        
        # Get discriminator score for GAN component
        disc_score = self.discriminator(x_hat)
        
        # Process through backbones
        x1 = self.convnext_backbone(x)
        x2 = self.convnext_backbone(x_hat)
        
        # Concatenate features
        x = torch.cat((x1, x2), dim=1)
        
        # Process through FC layers
        x = self.fc2(self.fc(x))
        
        # Calculate contrastive loss if targets are provided
        contrastive_loss = 0
        if target is not None and self.training:
            # Separate real and fake samples based on target
            real_indices = (target == 0).nonzero(as_tuple=True)[0]
            fake_indices = (target == 1).nonzero(as_tuple=True)[0]
            
            if len(real_indices) > 0 and len(fake_indices) > 0:
                z_real = mu[real_indices]
                z_fake = mu[fake_indices]
                contrastive_loss = self.contrastive_loss(z_real, z_fake) * self.contrastive_weight
        
        return x, self.resize(x_hat), disc_score, contrastive_loss
