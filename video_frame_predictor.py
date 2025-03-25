# import cv2
# import os
# import torch
# import torchvision.transforms as transforms
# from torchvision.io import read_image
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# import argparse
# from model.genconvit_vae import GenConViTVAE
# from model.config import load_config
# config = load_config()
# # Ensure CUDA usage if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # --- Frame Extraction and Preprocessing ---
# def extract_frames(video_path, frame_rate=1):
#     """Extract frames from a video at the specified frame rate."""
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     frame_interval = max(int(fps / frame_rate), 1)

#     for i in range(total_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if i % frame_interval == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)

#     cap.release()
#     return frames


# def preprocess_frame(frame):
#     """Preprocess a single frame to match the model's input dimensions."""
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])
    
#     frame_tensor = transform(frame).unsqueeze(0)
#     return frame_tensor


# # --- Model Prediction ---
# def load_model(model_path, device):
#     """Load the trained model with appropriate state dictionary handling."""
#     model = GenConViTVAE(config).to(device)
    
#     # Load model weights safely
#     checkpoint = torch.load(model_path, map_location=device)
    
#     if 'state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['state_dict'])
#     else:
#         model.load_state_dict(checkpoint)

#     model.eval()
#     return model


# def predict_frame(model, frame_tensor, device):
#     """Perform inference on a single frame and return the prediction."""
#     model.eval()
#     with torch.no_grad():
#         outputs = model(frame_tensor.to(device))
        
#         # Handle tuple outputs by selecting the main prediction tensor
#         main_output = outputs[0] if isinstance(outputs, tuple) else outputs
        
#         # Apply torch.max only on the main tensor
#         _, predicted = torch.max(main_output, 1)
    
#     return predicted.item()


# # --- Video Processing ---
# def process_video(video_path, model, device, frame_rate):
#     """Process a single video and return predictions for all frames."""
#     frames = extract_frames(video_path, frame_rate)
#     predictions = []

#     for frame in frames:
#         frame_tensor = preprocess_frame(frame)
#         prediction = predict_frame(model, frame_tensor, device)
#         predictions.append(prediction)
    
#     return predictions


# def process_videos_in_directory(directory, model_path, output_file, frame_rate=1):
#     """Process all videos in a directory and save predictions to an output file."""
#     model = load_model(model_path, device)
    
#     with open(output_file, "w") as out_file:
#         out_file.write("Video,Frame,Prediction\n")

#         video_files = [f for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mov'))]

#         for video_file in tqdm(video_files, desc="Processing videos"):
#             video_path = os.path.join(directory, video_file)
            
#             try:
#                 predictions = process_video(video_path, model, device, frame_rate)
                
#                 for frame_idx, prediction in enumerate(predictions):
#                     out_file.write(f"{video_file},{frame_idx},{prediction}\n")

#             except Exception as e:
#                 print(f"Error processing {video_file}: {str(e)}")


# # --- Command-line Interface ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process video frames with GenConViT model")
#     parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing videos")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights (.pth file)")
#     parser.add_argument("--output_file", type=str, required=True, help="Path to save predictions")
#     parser.add_argument("--frame_rate", type=int, default=1, help="Number of frames per second to process (default: 1)")

#     args = parser.parse_args()

#     process_videos_in_directory(args.directory, args.model_path, args.output_file, args.frame_rate)
import cv2
import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from collections import Counter
from model.genconvit_vae import GenConViTVAE
from model.config import load_config

# Load configuration
config = load_config()

# Ensure CUDA usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Frame Extraction and Preprocessing ---
def extract_frames(video_path, frame_rate=1):
    """Extract frames from a video at the specified frame rate."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_interval = max(int(fps / frame_rate), 1)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


def preprocess_frame(frame):
    """Preprocess a single frame to match the model's input dimensions."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    frame_tensor = transform(frame).unsqueeze(0)
    return frame_tensor


# --- Model Prediction ---
def load_model(model_path, device):
    """Load the trained model with appropriate state dictionary handling."""
    model = GenConViTVAE(config).to(device)
    
    # Load model weights safely
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def predict_frame(model, frame_tensor, device):
    """Perform inference on a single frame and return the prediction."""
    model.eval()
    with torch.no_grad():
        outputs = model(frame_tensor.to(device))
        
        # Handle tuple outputs by selecting the main prediction tensor
        main_output = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Apply torch.max only on the main tensor
        _, predicted = torch.max(main_output, 1)
    
    return predicted.item()


# --- Video Processing ---
def process_video(video_path, model, device, frame_rate):
    """Process a single video and return predictions for all frames."""
    frames = extract_frames(video_path, frame_rate)
    predictions = []

    for frame in frames:
        frame_tensor = preprocess_frame(frame)
        prediction = predict_frame(model, frame_tensor, device)
        predictions.append(prediction)
    
    return predictions


def process_videos_in_directory(directory, model_path, output_file, frame_rate=1):
    """Process all videos in a directory and save predictions to an output file."""
    model = load_model(model_path, device)
    
    with open(output_file, "w") as out_file:
        out_file.write("Video,Frame,Prediction,Overall_Prediction\n")

        video_files = [f for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mov'))]

        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(directory, video_file)
            
            try:
                predictions = process_video(video_path, model, device, frame_rate)

                # Determine overall prediction by most common label
                overall_prediction = Counter(predictions).most_common(1)[0][0]

                # Write predictions frame by frame
                for frame_idx, prediction in enumerate(predictions):
                    out_file.write(f"{video_file},{frame_idx},{prediction},{overall_prediction}\n")

                # Print the overall prediction for the video
                print(f"Overall Prediction for {video_file}: {overall_prediction}")

            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")


# --- Command-line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames with GenConViT model")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing videos")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights (.pth file)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--frame_rate", type=int, default=1, help="Number of frames per second to process (default: 1)")

    args = parser.parse_args()

    process_videos_in_directory(args.directory, args.model_path, args.output_file, args.frame_rate)
