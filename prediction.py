import os
import argparse
import json
from time import perf_counter
from datetime import datetime
import torch
from torch.cuda.amp import autocast
from model.pred_func import *
from model.config import load_config

# Load configuration
config = load_config()
print("CONFIG LOADED")
print(config)

# ---------------------------------------
# 🛠️ Prediction Function
# ---------------------------------------
def predict(
    vid, model, fp16, result, num_frames, net, klass, count=0, accuracy=-1, correct_label="unknown", compression=None
):
    """ Perform prediction on a video file """
    count += 1
    print(f"\n\n{str(count)} Loading... {vid}")

    # Extract faces from frames
    df = df_face(vid, num_frames, net)

    if len(df) < 1:
        print(f"No faces detected in {vid}. Skipping...")
        return result, accuracy, count, [0, 0.5]

    # Mixed precision inference
    with torch.no_grad():
        with autocast():  
            if fp16:
                df = df.half()

            y, y_val = pred_vid(df, model)

    # Store results
    result = store_result(result, os.path.basename(vid), y, y_val, klass, correct_label, compression)

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1

        print(f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} ({accuracy/count:.2%})")

    # Clear memory cache
    torch.cuda.empty_cache()
    del df, y, y_val

    return result, accuracy, count, [y, y_val]

# ---------------------------------------
# 📹 Video Processing Functions
# ---------------------------------------
def process_videos(ed_weight, vae_weight, root_dir, dataset, num_frames, net, fp16, dataset_func):
    """ Generalized function to process videos """
    result = set_result()
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    count, accuracy = 0, 0

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)

        try:
            if is_video(curr_vid):
                result, accuracy, count, _ = predict(
                    curr_vid, model, fp16, result, num_frames, net, "uncategorized", count, accuracy
                )
            else:
                print(f"Invalid video file: {curr_vid}. Skipping...")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result

# ---------------------------------------
# 🛠️ Dataset-specific Functions
# ---------------------------------------
def faceforensics(ed_weight, vae_weight, root_dir, num_frames, net, fp16):
    """ Process FaceForensics dataset """
    vid_types = ["original_sequences", "manipulated_sequences"]
    result = set_result()
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    with open(os.path.join("json_file", "ff_file_list.json")) as j_file:
        ff_file = json.load(j_file)

    count, accuracy = 0, 0

    for v_t in vid_types:
        for dirpath, _, filenames in os.walk(os.path.join(root_dir, v_t)):
            for filename in filenames:
                if filename in ff_file:
                    curr_vid = os.path.join(dirpath, filename)

                    try:
                        if is_video(curr_vid):
                            result, accuracy, count, _ = predict(
                                curr_vid, model, fp16, result, num_frames, net, "faceforensics", count, accuracy
                            )
                        else:
                            print(f"Invalid video file: {curr_vid}. Skipping...")

                    except Exception as e:
                        print(f"An error occurred: {str(e)}")

    return result


def dfdc(ed_weight, vae_weight, root_dir, num_frames, net, fp16):
    """ Process DFDC dataset """
    result = set_result()

    if os.path.isfile(os.path.join("json_file", "dfdc_files.json")):
        with open(os.path.join("json_file", "dfdc_files.json")) as data_file:
            dfdc_data = json.load(data_file)

    if os.path.isfile(os.path.join(root_dir, "metadata.json")):
        with open(os.path.join(root_dir, "metadata.json")) as data_file:
            dfdc_meta = json.load(data_file)

    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    count, accuracy = 0, 0

    for dfdc in dfdc_data:
        dfdc_file = os.path.join(root_dir, dfdc)

        try:
            if is_video(dfdc_file):
                label = dfdc_meta[dfdc]["label"]
                result, accuracy, count, _ = predict(
                    dfdc_file, model, fp16, result, num_frames, net, "dfdc", count, accuracy, label
                )
            else:
                print(f"Invalid video file: {dfdc_file}. Skipping...")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result


def celeb(ed_weight, vae_weight, root_dir, num_frames, net, fp16):
    """ Process Celeb-DF dataset """
    with open(os.path.join("json_file", "celeb_test.json"), "r") as f:
        celeb_files = json.load(f)

    result = set_result()
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    count, accuracy = 0, 0

    for ck in celeb_files:
        klass, filename = ck.split("/")
        label = "FAKE" if klass == "Celeb-synthesis" else "REAL"
        vid = os.path.join(root_dir, ck)

        try:
            if is_video(vid):
                result, accuracy, count, _ = predict(
                    vid, model, fp16, result, num_frames, net, klass, count, accuracy, label
                )
            else:
                print(f"Invalid video file: {vid}. Skipping...")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result

# ---------------------------------------
# 🛠️ Argument Parser
# ---------------------------------------
def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction")

    parser.add_argument("--p", type=str, help="video or image path")
    parser.add_argument("--f", type=int, help="number of frames to process", default=15)
    parser.add_argument("--d", type=str, help="dataset type: dfdc, faceforensics, celeb")
    parser.add_argument("--s", help="model size type: tiny, large.")
    parser.add_argument("--e", default='genconvit_ed_inference', help="weight for ed.")
    parser.add_argument("--v", default='genconvit_vae_inference', help="weight for vae.")
    parser.add_argument("--fp16", action="store_true", help="enable half precision")

    args = parser.parse_args()
    
    return args

# ---------------------------------------
# 🚀 Main Execution Function
# ---------------------------------------
def main():
    start_time = perf_counter()

    args = gen_parser()
    config["model"]["backbone"] = f"convnext_{args.s}" if args.s else "convnext_tiny"
    
    # Load dataset processor
    dataset_func = globals().get(args.d, process_videos)

    # Perform prediction
    result = dataset_func(args.e, args.v, args.p, args.f, "vae", args.fp16)

    # Save results
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"result/prediction_{args.d}_{curr_time}.json"

    with open(file_path, "w") as f:
        json.dump(result, f)

    end_time = perf_counter()
    print(f"\n--- Completed in {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
