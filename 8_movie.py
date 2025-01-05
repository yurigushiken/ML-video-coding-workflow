#!/usr/bin/env python3

import os
import csv
import subprocess
import re
import time

# ============================================
# TOGGLE SETTINGS FOR TEXT OVERLAYS
# ============================================
SHOW_LEFT_OVERLAY = True   # If False, hides frame # and "What, Where"
SHOW_RIGHT_OVERLAY = True  # If False, hides "time, events_corrected, trial_number, segment"
SHOW_BOTTOM_OVERLAY = True # If False, hides "participant type & age"

# A serif font (adjust if needed)
FONT_PATH_SERIF = r"/path/to/times.ttf"

def escape_drawtext(text):
    """
    Escapes certain special characters so ffmpeg's drawtext won't cut off or fail.
    - Colons (:) need escaping or FFmpeg treats them like parameter separators.
    - Single quotes (') also need escaping.
    """
    if not text:
        return ""
    text = text.replace(':', '\\:')
    text = text.replace("'", "\\'")
    return text

def find_csv_in_datasheet_dir(datasheet_dir):
    for fname in os.listdir(datasheet_dir):
        if fname.lower().endswith(".csv"):
            return os.path.join(datasheet_dir, fname)
    return None

def find_datasheet_dirs(base_path):
    datasheet_dirs = []
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root).lower() == "datasheet":
            datasheet_dirs.append(root)
    return datasheet_dirs

def frame_to_timecode(frame_num, fps=30):
    total_seconds = frame_num / float(fps)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = (total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

def find_participant_csv(participant_dir):
    """
    Finds the single CSV in participant_dir (not in the 'datasheet' folder).
    Returns the absolute path or None if none is found.
    """
    csv_candidates = []
    for fname in os.listdir(participant_dir):
        if fname.lower().endswith(".csv"):
            full_path = os.path.join(participant_dir, fname)
            if os.path.isfile(full_path):
                if "datasheet" not in full_path.lower():
                    csv_candidates.append(full_path)

    if not csv_candidates:
        return None
    return csv_candidates[0]

def read_merged_data(participant_dir):
    """
    Merges data from the datasheet CSV (in participant_dir/datasheet/)
    and participant CSV (in participant_dir).

    Returns:
      csv_data_by_frame: dict {frame_num: row_data}
      participant_type: str
      participant_age_months: str
      participant_age_years: str
    """
    datasheet_subdir = os.path.join(participant_dir, "datasheet")
    if not os.path.isdir(datasheet_subdir):
        raise FileNotFoundError(f"No datasheet subfolder found in {participant_dir}")

    datasheet_csv = find_csv_in_datasheet_dir(datasheet_subdir)
    if not datasheet_csv:
        raise FileNotFoundError(f"No CSV found in datasheet subfolder: {datasheet_subdir}")

    datasheet_data = {}
    with open(datasheet_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if "Frame Number" not in reader.fieldnames:
            raise ValueError(f"CSV '{datasheet_csv}' must contain 'Frame Number' column.")
        for row in reader:
            frame_str = row.get("Frame Number", "").strip()
            if frame_str.isdigit():
                datasheet_data[int(frame_str)] = row

    participant_csv = find_participant_csv(participant_dir)

    participant_data = {}
    participant_type = None
    participant_age_months = None
    participant_age_years = None

    if participant_csv and os.path.exists(participant_csv):
        with open(participant_csv, newline='', encoding='utf-8') as f:
            p_reader = csv.DictReader(f)
            for p_row in p_reader:
                frame_str_p = p_row.get("Frame Number", "").strip()
                if frame_str_p.isdigit():
                    participant_data[int(frame_str_p)] = p_row

                if participant_type is None and "participant_type" in p_row:
                    participant_type = p_row["participant_type"].strip()
                if participant_age_months is None and "participant_age_months" in p_row:
                    participant_age_months = p_row["participant_age_months"].strip()
                if participant_age_years is None and "participant_age_years" in p_row:
                    participant_age_years = p_row["participant_age_years"].strip()
    else:
        print(f"No participant CSV found in {participant_dir}. Only using datasheet CSV data.")

    csv_data_by_frame = {}
    all_frame_nums = set(datasheet_data.keys()).union(participant_data.keys())
    for fnum in all_frame_nums:
        merged_row = {}
        if fnum in datasheet_data:
            merged_row.update(datasheet_data[fnum])
        if fnum in participant_data:
            merged_row.update(participant_data[fnum])
        merged_row["Frame Number"] = fnum
        csv_data_by_frame[fnum] = merged_row

    return csv_data_by_frame, participant_type, participant_age_months, participant_age_years

def gather_frame_numbers_in_visual_outputs(visual_outputs_dir):
    """
    Looks for files named 'frame_XXXX_annotated.jpg' and extracts XXXX as an integer.
    Returns a sorted list of all discovered frame numbers.
    """
    frame_nums = []
    pattern = re.compile(r"^frame_(\d+)_annotated\.jpg$", re.IGNORECASE)
    for fname in os.listdir(visual_outputs_dir):
        match = pattern.match(fname)
        if match:
            frame_nums.append(int(match.group(1)))
    return sorted(frame_nums)

def process_datasheet(datasheet_dir, overall_start_time):
    participant_dir = os.path.dirname(datasheet_dir)
    MOVIE_FRAMES_DIR = os.path.join(participant_dir, "inference_output", "movie frames")

    avi_files = [f for f in os.listdir(participant_dir) if f.lower().endswith('.avi')]
    if avi_files:
        avi_file_base = os.path.splitext(avi_files[0])[0]
        final_movie_name = f"{avi_file_base}-movie.mp4"
    else:
        final_movie_name = "movie.mp4"
    mp4_output = os.path.join(participant_dir, final_movie_name)

    if os.path.exists(mp4_output):
        print(f"Skipping {participant_dir}. The final movie '{final_movie_name}' already exists.\n")
        return

    csv_data_by_frame, participant_type, participant_age_months, participant_age_years = read_merged_data(participant_dir)

    VISUAL_OUTPUTS_DIR = os.path.join(participant_dir, "inference_output", "visual_outputs")
    if not os.path.isdir(VISUAL_OUTPUTS_DIR):
        print(f"No 'visual_outputs' folder found: {VISUAL_OUTPUTS_DIR}. Skipping.")
        return
    all_frame_nums = gather_frame_numbers_in_visual_outputs(VISUAL_OUTPUTS_DIR)
    if not all_frame_nums:
        print(f"No annotated frames found in {VISUAL_OUTPUTS_DIR}. Skipping.")
        return

    os.makedirs(MOVIE_FRAMES_DIR, exist_ok=True)
    max_frame_num = max(all_frame_nums)
    total_frames = len(all_frame_nums)

    start_time_dir = time.time()
    last_messages = []

    def print_dynamic_status(current_frame_index, total_frames):
        print("\033[H\033[J", end="")
        elapsed_dir = time.time() - start_time_dir
        elapsed_total = time.time() - overall_start_time
        if current_frame_index > 0:
            est_total_dir = (elapsed_dir / current_frame_index) * total_frames
            remain_dir = est_total_dir - elapsed_dir
        else:
            est_total_dir = 0
            remain_dir = 0

        print(f"Currently processing directory: {datasheet_dir}")
        print(f"Frame progress: {current_frame_index}/{total_frames}")
        print(f"Elapsed (this directory): {time.strftime('%H:%M:%S', time.gmtime(elapsed_dir))}")
        print(f"Estimated total (this directory): {time.strftime('%H:%M:%S', time.gmtime(est_total_dir))}")
        print(f"Estimated remaining (this directory): {time.strftime('%H:%M:%S', time.gmtime(remain_dir))}")
        print(f"Total elapsed (whole script): {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print()

        for msg in last_messages[-10:]:
            print(msg)

    print(f"Generating overlays in '{MOVIE_FRAMES_DIR}' (if missing).")

    for i, fnum in enumerate(all_frame_nums, start=1):
        out_filename = f"frame_{fnum:04d}.png"
        out_path = os.path.join(MOVIE_FRAMES_DIR, out_filename)

        if os.path.exists(out_path):
            progress_msg = f"Frame {fnum:04d} already exists. Skipping overlay."
            last_messages.append(progress_msg)
            print_dynamic_status(i, total_frames)
            continue

        image_name = f"frame_{fnum:04d}_annotated.jpg"
        image_path = os.path.join(VISUAL_OUTPUTS_DIR, image_name)
        if not os.path.exists(image_path):
            progress_msg = f"Missing annotated image for frame {fnum:04d}; skipping."
            last_messages.append(progress_msg)
            print_dynamic_status(i, total_frames)
            continue

        row = csv_data_by_frame.get(fnum, {})
        filter_str = "crop=900:700:(in_w-900)/2:(in_h-700)/2"

        if SHOW_LEFT_OVERLAY:
            frame_label = f"{fnum}/{max_frame_num}"
            what_value = row.get("What", "").strip().lower()
            where_value = row.get("Where", "").strip().lower()
            top_text = f"{what_value}, {where_value}"

            frame_label_esc = escape_drawtext(frame_label)
            top_text_esc = escape_drawtext(top_text)

            filter_str += (
                f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                f"text='{frame_label_esc}':x=20:y=20:"
                f"fontcolor=white:fontsize=24,"
                f"drawtext=fontfile='{FONT_PATH_SERIF}':text='{top_text_esc}':"
                f"x=(w-text_w)/2:y=40:fontcolor=white:fontsize=72"
            )

        if SHOW_RIGHT_OVERLAY:
            time_str = frame_to_timecode(fnum, 30)
            events_corrected = row.get("events_corrected", "").strip()
            trial_number = row.get("trial_number", "").strip()
            segment = row.get("segment", "").strip()
            frame_count_segment = row.get("frame_count_segment", "").strip()

            events_corrected_and_trial = f"{events_corrected} {trial_number}".strip()

            time_str_esc = escape_drawtext(time_str)
            events_corrected_esc = escape_drawtext(events_corrected_and_trial)
            segment_esc = escape_drawtext(segment + " " + frame_count_segment)

            filter_str += (
                f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                f"text='{time_str_esc}':x=(w-text_w)-20:y=20:"
                f"fontcolor=white:fontsize=24,"
                f"drawtext=fontfile='{FONT_PATH_SERIF}':"
                f"text='{events_corrected_esc}':"
                f"x=(w-text_w)-20:y=50:fontcolor=white:fontsize=24,"
                f"drawtext=fontfile='{FONT_PATH_SERIF}':"
                f"text='{segment_esc}':"
                f"x=(w-text_w)-20:y=80:fontcolor=white:fontsize=24"
            )

        if SHOW_BOTTOM_OVERLAY and participant_type:
            participant_type_esc = escape_drawtext(participant_type.lower())
            if participant_type.lower() == "adult" and participant_age_years:
                age_str = f"{participant_age_years} years"
            elif participant_age_months:
                age_str = f"{participant_age_months} months"
            else:
                age_str = "(age unknown)"

            participant_text = f"participant: {participant_type_esc} {age_str}"
            participant_text_esc = escape_drawtext(participant_text)

            filter_str += (
                f",drawtext=fontfile='{FONT_PATH_SERIF}':"
                f"text='{participant_text_esc}':"
                f"x=(w-text_w)/2:y=(h-text_h)-60:"
                f"fontcolor=white:fontsize=28"
            )

        bottom_right_text = escape_drawtext("errors corrected with rules")
        filter_str += (
            f",drawtext=fontfile='{FONT_PATH_SERIF}':"
            f"text='{bottom_right_text}':"
            f"x=(w-text_w)-20:y=(h-text_h)-20:"
            f"fontcolor=white:fontsize=16"
        )

        progress_msg = f"Processing frame {fnum:04d} -> {out_filename}"
        last_messages.append(progress_msg)
        print_dynamic_status(i, total_frames)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-y",
            "-i", image_path,
            "-vf", filter_str,
            out_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print()
    combination_msg = "Combining frames into final movie..."
    last_messages.append(combination_msg)
    print_dynamic_status(total_frames, total_frames)

    cmd_combine = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-framerate", "30",
        "-i", os.path.join(MOVIE_FRAMES_DIR, "frame_%04d.png"),
        "-pix_fmt", "yuv420p",
        "-vcodec", "libx264",
        mp4_output
    ]
    subprocess.run(cmd_combine, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    done_msg = f"Movie saved to: {mp4_output}\n"
    last_messages.append(done_msg)
    print_dynamic_status(total_frames, total_frames)

def main():
    overall_start_time = time.time()
    base_path = r"/path/to/base_directory/Adult"
    datasheet_dirs = find_datasheet_dirs(base_path)
    if not datasheet_dirs:
        raise FileNotFoundError("No folder named 'datasheet' found in the directory structure.")

    for ds_dir in datasheet_dirs:
        process_datasheet(ds_dir, overall_start_time)

if __name__ == "__main__":
    main()
