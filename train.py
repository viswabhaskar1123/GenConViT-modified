import sys, os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from time import perf_counter
import pickle
from model.config import load_config
from model.genconvit_ed import GenConViTED
from model.genconvit_vae import GenConViTVAE
from dataset.loader import load_data, load_checkpoint
import optparse

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained(pretrained_model_filename):
    assert os.path.isfile(
        pretrained_model_filename
    ), "Saved model file does not exist. Exiting."

    model, optimizer, start_epoch, min_loss = load_checkpoint(
        model, optimizer, filename=pretrained_model_filename
    )
    # now individually transfer the optimizer parts...
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return model, optimizer, start_epoch, min_loss


def train_model(
    dir_path, mod, num_epochs, pretrained_model_filename, test_model, batch_size
):
    print("Loading data...")
    dataloaders, dataset_sizes = load_data(dir_path, batch_size)
    print("Done.")

    if mod == "ed":
        from train.train_ed import train, valid
        model = GenConViTED(config)
    else:
        from train.train_vae import train, valid
        model = GenConViTVAE(config)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    mse = nn.MSELoss()
    min_val_loss = int(config["min_val_loss"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    if pretrained_model_filename:
        model, optimizer, start_epoch, min_loss = load_pretrained(
            pretrained_model_filename
        )

    model.to(device)
    torch.manual_seed(1)
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    since = time.time()

    for epoch in range(0, num_epochs):
        train_loss, train_acc, epoch_loss = train(
            model,
            device,
            dataloaders["train"],
            criterion,
            optimizer,
            epoch,
            train_loss,
            train_acc,
            mse,
        )
        valid_loss, valid_acc = valid(
            model,
            device,
            dataloaders["validation"],
            criterion,
            epoch,
            valid_loss,
            valid_acc,
            mse,
        )
        scheduler.step()

    time_elapsed = time.time() - since

    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    print("\nSaving model...\n")
    # Define the save path in Google Drive
    drive_path = "/content/drive/My Drive/GenConViT_Models"
    os.makedirs(drive_path, exist_ok=True)
    
    file_name = f'genconvit_{mod}_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}'
    file_path = os.path.join(drive_path, file_name)
    
    # Save training history
    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump([train_loss, train_acc, valid_loss, valid_acc], f)
    
    # Save model state
    state = {
        "epoch": num_epochs + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "min_loss": epoch_loss,
    }
    
    weight_path = f"{file_path}.pth"
    torch.save(state, weight_path)
    
    print(f"✅ Model saved successfully to: {weight_path}")
    file_path = os.path.join(
        "weight",
        f'genconvit_{mod}_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}',
    )

    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump([train_loss, train_acc, valid_loss, valid_acc], f)

    state = {
        "epoch": num_epochs + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "min_loss": epoch_loss,
    }

    weight = f"{file_path}.pth"
    torch.save(state, weight)

    print("Done.")

    if test_model:
        test(model, dataloaders, dataset_sizes, mod, weight)

# import sys, os
# import numpy as np
# import torch
# from torch import nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import time
# from time import perf_counter
# import pickle
# from model.config import load_config
# from model.genconvit_v2 import GenConViTV2  # Updated model import
# from dataset.loader import load_data, load_checkpoint
# import optparse

# config = load_config()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def load_pretrained(pretrained_model_filename):
#     assert os.path.isfile(pretrained_model_filename), "Model file not found."
#     model, optimizer, start_epoch, min_loss = load_checkpoint(model, optimizer, filename=pretrained_model_filename)

#     # Transfer optimizer to GPU
#     for state in optimizer.state.values():
#         for k, v in state.items():
#             if isinstance(v, torch.Tensor):
#                 state[k] = v.to(device)
#     return model, optimizer, start_epoch, min_loss



# # Mixed precision scaler
# scaler = torch.cuda.amp.GradScaler()

# def train_model(dir_path, mod, num_epochs, pretrained_model_filename, test_model, batch_size):
#     print("Loading data...")
#     dataloaders, dataset_sizes = load_data(dir_path, batch_size)
#     print("Done.")

#     model = GenConViTV2(config).to(device)  # Use updated model

#     # Improved optimizer with Lookahead and Ranger
#     base_optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
#     optimizer = optim.Ranger(base_optimizer, k=5, alpha=0.5)

#     criterion = nn.CrossEntropyLoss()
#     focal_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.75, 1.25]).to(device))

#     mse = nn.MSELoss()
#     min_val_loss = int(config["min_val_loss"])
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

#     criterion = nn.CrossEntropyLoss()

#     if pretrained_model_filename:
#         model, optimizer, start_epoch, min_loss = load_pretrained(pretrained_model_filename)

#     model.to(device)
#     torch.manual_seed(1)

#     train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

#     since = time.time()

#     for epoch in range(0, num_epochs):
#         print(f"Epoch {epoch + 1}/{num_epochs}")

#         # Train phase
#         model.train()
#         running_loss, running_corrects = 0.0, 0

#         for inputs, labels in dataloaders["train"]:
#             inputs, labels = inputs.to(device), labels.to(device)

#             # MixUp Augmentation
#             lam = np.random.beta(0.4, 0.4)
#             inputs, targets_a, targets_b = inputs, labels, labels[torch.randperm(labels.size(0))]

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)

#         scheduler.step()

#         time_elapsed = time.time() - since
#         print(f"Epoch {epoch + 1} completed in {time_elapsed:.2f}s")

#     # Save model
#     file_path = os.path.join("weight", f'genconvit_v2_{time.strftime("%Y_%m_%d_%H_%M_%S")}.pth')
#     torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, file_path)

#     print("Training Complete!")


import torch
import numpy as np
from torch.cuda.amp import autocast

def test(model, dataloaders, dataset_sizes, mod, weight):
    print("\nRunning test...\n")
    
    # Clear cache before testing
    torch.cuda.empty_cache()

    model.eval()

    # Load the checkpoint and weights
    checkpoint = torch.load(weight, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        with autocast():  # Use mixed precision
            for batch_idx, (inputs, labels) in enumerate(dataloaders["test"]):
                inputs, labels = inputs.to(device), labels.to(device)

                # Inference with or without encoder-decoder mode
                if mod == "ed":
                    output = model(inputs).to(device).float()
                else:
                    output, _, _ = model(inputs)  # Assuming model returns (output, recons, kl_div)
                    output = output.to(device).float()

                # Get predictions
                _, predictions = torch.max(output, 1)

                # Compare predictions with labels
                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

                # Display progress every 10 batches
                if batch_idx % 10 == 0:
                    accuracy = (total_correct / total_samples) * 100
                    print(f"Batch {batch_idx}/{len(dataloaders['test'])} - Accuracy: {accuracy:.2f}%")

    # Final accuracy calculation
    final_accuracy = (total_correct / dataset_sizes["test"]) * 100
    print(f'\nFinal Accuracy: {total_correct}/{dataset_sizes["test"]} ({final_accuracy:.2f}%)\n')

    # Free memory manually after testing
    del inputs, labels, output, predictions
    torch.cuda.empty_cache()


def gen_parser():
    parser = optparse.OptionParser("Train GenConViT model.")
    parser.add_option(
        "-e",
        "--epoch",
        type=int,
        dest="epoch",
        help="Number of epochs used for training the GenConvNextViT model.",
    )
    parser.add_option("-v", "--version", dest="version", help="Version 0.1.")
    parser.add_option("-d", "--dir", dest="dir", help="Training data path.")
    parser.add_option(
        "-m",
        "--model",
        dest="model",
        help="model ed or model vae, model variant: genconvit (A) ed or genconvit (B) vae.",
    )
    parser.add_option(
        "-p",
        "--pretrained",
        dest="pretrained",
        help="Saved model file name. If you want to continue from the previous trained model.",
    )
    parser.add_option("-t", "--test", dest="test", help="run test on test dataset.")
    parser.add_option("-b", "--batch_size", dest="batch_size", help="batch size.")

    (options, _) = parser.parse_args()

    dir_path = options.dir
    epoch = options.epoch
    mod = "ed" if options.model == "ed" else "vae"
    test_model = "y" if options.test else None
    pretrained_model_filename = options.pretrained if options.pretrained else None
    batch_size = options.batch_size if options.batch_size else config["batch_size"]

    return dir_path, mod, epoch, pretrained_model_filename, test_model, int(batch_size)


def main():
    start_time = perf_counter()
    path, mod, epoch, pretrained_model_filename, test_model, batch_size = gen_parser()
    train_model(path, mod, epoch, pretrained_model_filename, test_model, batch_size)
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
