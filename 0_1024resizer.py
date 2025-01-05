#!/usr/bin/env python3
# 1024resizer.py

import os
import re
import subprocess
import sys

# IMPORTANT: Update parent_dir using forward slashes:
parent_dir = r"/path/to/data_processing"

def sanitize_name(name):
    """
    Removes parentheses and their contents from the name,
    strips out spaces, and makes it lowercase for simpler matching.
    """
    name_no_paren = re.sub(r"\([^)]*\)", "", name)
    name_clean = name_no_paren.replace(" ", "").lower()
    return name_clean

def create_1024_frames(video_path, output_dir):
    """
    Checks if the 1024x1024_frames directory already exists.
    If it does, we skip this video. If not, we create it
    and process the frames via ffmpeg.
    """
    if os.path.isdir(output_dir):
        print(f"Skipping (already processed): {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    msg = f"Processing {video_path}"
    print(f"\r{msg}", end="", flush=True)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "pad=1024:1024:0:128:black",
        os.path.join(output_dir, "frame_%04d.png")
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    final_msg = f"Done processing {video_path}"
    print(f"\r{final_msg}{' ' * (len(msg) - len(final_msg))}")

def main():
    for root, dirs, files in os.walk(parent_dir):
        folder_name = os.path.basename(root)
        folder_sanitized = sanitize_name(folder_name)

        for f in files:
            if f.lower().endswith(".avi"):
                file_sanitized = sanitize_name(os.path.splitext(f)[0])
                if folder_sanitized in file_sanitized or file_sanitized in folder_sanitized:
                    video_path = os.path.join(root, f)
                    output_dir = os.path.join(root, "1024x1024_frames")
                    create_1024_frames(video_path, output_dir)

if __name__ == "__main__":
    main()


