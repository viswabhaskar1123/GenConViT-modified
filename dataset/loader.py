# original
import os
import torch
from torchvision import transforms, datasets
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    HueSaturationValue,
    GaussNoise,
    Sharpen,
    Emboss,
    RandomBrightnessContrast,
    OneOf,
    Compose,
)
import numpy as np
from PIL import Image


def strong_aug(p=0.5):
    return Compose(
        [
            RandomRotate90(p=0.2),
            Transpose(p=0.2),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            OneOf(
                [
                    GaussNoise(),
                ],
                p=0.2,
            ),
            ShiftScaleRotate(p=0.2),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    Sharpen(),
                    Emboss(),
                    RandomBrightnessContrast(),
                ],
                p=0.2,
            ),
            HueSaturationValue(p=0.2),
        ],
        p=p,
    )


def augment(aug, image):
    return aug(image=image)["image"]


class Aug(object):
    def __call__(self, img):
        aug = strong_aug(p=0.9)
        return Image.fromarray(augment(aug, np.array(img)))


def normalize_data():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return {
        "train": transforms.Compose(
            [Aug(), transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
        "valid": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
        "test": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
        "vid": transforms.Compose([transforms.Normalize(mean, std)]),
    }


def load_data(data_dir="sample/", batch_size=4):
    data_dir = data_dir
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), normalize_data()[x])
        for x in ["train", "valid", "test"]
    }

    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,
    #                                             shuffle=True, num_workers=0, pin_memory=True)
    #               for x in ['train', 'validation', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid", "test"]}

    train_dataloaders = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    validation_dataloaders = torch.utils.data.DataLoader(
        image_datasets["valid"],
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_dataloaders = torch.utils.data.DataLoader(
        image_datasets["test"],
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    dataloaders = {
        "train": train_dataloaders,
        "validation": validation_dataloaders,
        "test": test_dataloaders,
    }

    return dataloaders, dataset_sizes


def load_checkpoint(model, optimizer, filename=None):
    start_epoch = 0
    log_loss = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        log_loss = checkpoint["min_loss"]
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"])
        )
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, log_loss
#original
# import os
# import torch
# from torchvision import transforms, datasets
# from albumentations import (
#     HorizontalFlip,
#     VerticalFlip,
#     ShiftScaleRotate,
#     CLAHE,
#     RandomRotate90,
#     Transpose,
#     HueSaturationValue,
#     GaussNoise,
#     Sharpen,
#     Emboss,
#     RandomBrightnessContrast,
#     OneOf,
#     Compose
# )
# import numpy as np
# from PIL import Image

# ### 🛠️ Data Augmentation Pipeline ###
# def strong_aug(p=0.5):
#     """Strong Augmentation Pipeline using Albumentations."""
#     return Compose(
#         [
#             RandomRotate90(p=0.2),
#             Transpose(p=0.2),
#             HorizontalFlip(p=0.5),
#             VerticalFlip(p=0.5),
#             OneOf(
#                 [
#                     GaussNoise(var_limit=(10.0, 50.0), p=0.3),
#                 ],
#                 p=0.3,
#             ),
#             ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
#             OneOf(
#                 [
#                     CLAHE(clip_limit=2),
#                     Sharpen(alpha=(0.2, 0.5), p=0.3),
#                     Emboss(alpha=(0.2, 0.5), p=0.3),
#                     RandomBrightnessContrast(p=0.3),
#                 ],
#                 p=0.3,
#             ),
#             HueSaturationValue(p=0.2),
#         ],
#         p=p,
#     )


# ### 💡 Albumentations Wrapper for PyTorch ###
# class Aug(object):
#     """Wrapper class to apply Albumentations to PIL images with resizing."""

#     def __call__(self, img):
#         img = np.array(img)
        
#         # Apply augmentation
#         aug = strong_aug(p=0.9)
#         img = aug(image=img)["image"]
        
#         # Resize to uniform dimensions
#         img = Image.fromarray(img).resize((224, 224))  # Ensure all images are 224x224
#         return img


# ### 🔥 Normalization and Transformation ###
# def normalize_data():
#     """Return PyTorch transformations with augmentation applied for training."""
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]

#     return {
#         "train": transforms.Compose(
#             [
#                 Aug(),                                  # Apply augmentation
#                 transforms.Resize((224, 224)),           # Uniform resizing
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)
#             ]
#         ),
#         "valid": transforms.Compose(
#             [
#                 transforms.Resize((224, 224)),           # Ensure valid images are consistent
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)
#             ]
#         ),
#         "test": transforms.Compose(
#             [
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)
#             ]
#         ),
#         "vid": transforms.Compose(
#             [
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)
#             ]
#         )
#     }


# ### 🚀 Optimized DataLoader Function ###
# def load_data(data_dir="sample/", batch_size=16):
#     """Load datasets and create DataLoader with memory optimizations."""
    
#     # Data transformations
#     image_datasets = {
#         x: datasets.ImageFolder(os.path.join(data_dir, x), transform=normalize_data()[x])
#         for x in ["train", "valid", "test"]
#     }

#     # Use optimal num_workers and pin_memory for performance
#     dataloaders = {
#         "train": torch.utils.data.DataLoader(
#             image_datasets["train"],
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=4,              # Faster data loading
#             pin_memory=True
#         ),
#         "validation": torch.utils.data.DataLoader(
#             image_datasets["valid"],
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True
#         ),
#         "test": torch.utils.data.DataLoader(
#             image_datasets["test"],
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True
#         ),
#     }

#     dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid", "test"]}

#     print("Dataset sizes:", dataset_sizes)
    
#     return dataloaders, dataset_sizes


# ### 🔥 Improved Checkpoint Loader ###
# def load_checkpoint(model, optimizer, filename=None):
#     """Load model and optimizer from checkpoint with error handling."""
#     start_epoch = 0
#     log_loss = 0

#     if filename and os.path.isfile(filename):
#         print(f"Loading checkpoint from '{filename}'...")

#         # Handle potential loading errors gracefully
#         try:
#             checkpoint = torch.load(filename, map_location="cuda" if torch.cuda.is_available() else "cpu")
            
#             # Load state_dicts
#             model.load_state_dict(checkpoint["state_dict"])
#             optimizer.load_state_dict(checkpoint["optimizer"])

#             # Retrieve epoch and loss
#             start_epoch = checkpoint.get("epoch", 0)
#             log_loss = checkpoint.get("min_loss", 0)

#             print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        
#         except Exception as e:
#             print(f"Failed to load checkpoint: {e}")
    
#     else:
#         print(f"No checkpoint found at '{filename}'")

#     return model, optimizer, start_epoch, log_loss
