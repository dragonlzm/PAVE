# this script aims to select the avqa video from the vgg-sound subset

import os
import json
import shutil

def process_clips(json_file, folder_a, folder_b):
    # Load the JSON file
    with open(json_file, 'r') as f:
        video_data = json.load(f)

    # Ensure folder B exists
    if not os.path.exists(folder_b):
        os.makedirs(folder_b)

    # Iterate through the video IDs and timestamps
    for video_id, timestamps in video_data.items():
        for timestamp in timestamps:
            # Generate the clip name
            clip_name = f"{video_id}_{timestamp:06d}" + '.mp4'  # Assuming timestamp is an integer

            # Define the source and destination paths
            src_path = os.path.join(folder_a, clip_name)
            dest_path = os.path.join(folder_b, clip_name)

            # Check if the clip exists and copy it
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                print(f"Copied: {clip_name} to {folder_b}")
            else:
                print(f"Clip not found: {clip_name}")

# Example usage
json_file = "avqa_download_urls_mapping.json"  # Path to your JSON file
folder_a = "./scratch/shared/beegfs/hchen/train_data/VGGSound_final/video"   # Path to Folder A
folder_b = "./avqa_subset"   # Path to Folder B

process_clips(json_file, folder_a, folder_b)