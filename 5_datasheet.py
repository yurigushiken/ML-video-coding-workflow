import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

def create_mask_array(mask_pixels):
    """Creates a boolean mask array from mask pixels."""
    if not mask_pixels or len(mask_pixels) < 3:
        return None
    
    try:
        points = np.array(mask_pixels)
        height = int(np.max(points[:, 0])) + 1
        width = int(np.max(points[:, 1])) + 1
        mask = np.zeros((height, width), dtype=bool)
        y_coords = points[:, 0].astype(int)
        x_coords = points[:, 1].astype(int)
        mask[y_coords, x_coords] = True
        return mask
    except Exception as e:
        print(f"Error creating mask array: {e}")
        return None

def check_mask_intersection(mask1_pixels, mask2_pixels):
    """Check if two masks intersect by checking actual pixels."""
    if not mask1_pixels or not mask2_pixels:
        return False
    
    try:
        mask1 = create_mask_array(mask1_pixels)
        mask2 = create_mask_array(mask2_pixels)
        
        if mask1 is None or mask2 is None:
            return False
        
        max_height = max(mask1.shape[0], mask2.shape[0])
        max_width = max(mask1.shape[1], mask2.shape[1])
        
        if mask1.shape != (max_height, max_width):
            new_mask1 = np.zeros((max_height, max_width), dtype=bool)
            new_mask1[:mask1.shape[0], :mask1.shape[1]] = mask1
            mask1 = new_mask1
            
        if mask2.shape != (max_height, max_width):
            new_mask2 = np.zeros((max_height, max_width), dtype=bool)
            new_mask2[:mask2.shape[0], :mask2.shape[1]] = mask2
            mask2 = new_mask2
        
        intersection = np.logical_and(mask1, mask2)
        return np.any(intersection)
        
    except Exception as e:
        print(f"Error checking mask intersection: {e}")
        return False

def normalize_class_name(class_name):
    """Normalizes class names."""
    if not isinstance(class_name, str):
        return ""
    return class_name.strip().lower().replace(' ', '_')

def extract_frame_number(filename):
    """Extracts frame number from filename."""
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None

def calculate_time(frame_number, fps=30):
    """Converts frame number to time string."""
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    remainder = total_seconds % 3600
    minutes = int(remainder // 60)
    seconds = int(remainder % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
    return total_seconds, time_str

def determine_what_where(detections, priority_order):
    """Determines What and Where based on actual mask intersections."""
    if not detections or not isinstance(detections, list):
        return ('no', 'signal')
    
    try:
        blue_dots = [d for d in detections if normalize_class_name(d.get('class', '')) == 'blue_dot']
        if not blue_dots:
            return ('no', 'signal')
        
        blue_dot = max(blue_dots, key=lambda x: x.get('confidence', 0))
        blue_dot_mask = blue_dot.get('mask_pixels', [])
        
        if not blue_dot_mask:
            return ('no', 'signal')
        
        class_mapping = {
            'toy': ('toy', 'other'),
            'hand_man': ('man', 'hands'),
            'hand_woman': ('woman', 'hands'),
            'face_woman': ('woman', 'face'),
            'face_man': ('man', 'face'),
            'body_woman': ('woman', 'body'),
            'body_man': ('man', 'body'),
            'green_circle': ('green_circle', 'other'),
            'screen': ('screen', 'other')
        }
        
        for obj_class in priority_order:
            matching_objs = [
                d for d in detections 
                if normalize_class_name(d.get('class', '')) == normalize_class_name(obj_class)
            ]
            
            matching_objs.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            for obj in matching_objs:
                obj_mask = obj.get('mask_pixels', [])
                if not obj_mask:
                    continue
                
                if check_mask_intersection(blue_dot_mask, obj_mask):
                    key = normalize_class_name(obj.get('class', ''))
                    return class_mapping.get(key, (obj['class'].replace('_', ' '), 'other'))
        
        return ('screen', 'other')
        
    except Exception as e:
        print(f"Error in determine_what_where: {e}")
        return ('unknown', 'unknown')

def get_blue_dot_center(detections):
    """
    Returns the center (x, y) of the highest confidence blue_dot mask_pixels 
    as a string, e.g. '(123.45, 67.89)'. If none found, returns ''.
    """
    try:
        blue_dots = [
            d for d in detections 
            if normalize_class_name(d.get('class', '')) == 'blue_dot'
        ]
        if not blue_dots:
            return ''
        
        # Pick the blue dot with the highest confidence
        blue_dot = max(blue_dots, key=lambda x: x.get('confidence', 0))
        pixels = blue_dot.get('mask_pixels', [])
        if not pixels:
            return ''
        
        coords = np.array(pixels)
        y_vals = coords[:, 0]
        x_vals = coords[:, 1]
        center_x = np.mean(x_vals)
        center_y = np.mean(y_vals)
        return f"({center_x:.2f}, {center_y:.2f})"
    except:
        return ''

def process_detections(detections_dir, participant_dir, participant_id, fps=30):
    """Process all detection files in a directory."""
    # Create datasheet directory in participant's main folder
    output_dir = participant_dir / "datasheet"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the participant_id in the filename
    output_csv = output_dir / f"datasheet-{participant_id}.csv"

    # If a datasheet already exists for these detections, skip processing
    if output_csv.exists():
        print(f"Datasheet already exists for {participant_id} at {output_csv}, skipping processing.")
        return True

    # Initialize data list
    data = []

    # Define priority order
    priority_order = [
        'toy',
        'hand_man',
        'hand_woman',
        'face_woman',
        'face_man',
        'body_woman',
        'body_man',
        'green_circle',
        'screen'
    ]

    # Get JSON files
    json_files = sorted(
        detections_dir.glob("*.json"),
        key=lambda x: extract_frame_number(x.name) if extract_frame_number(x.name) is not None else -1
    )
    
    if not json_files:
        print(f"No JSON files found in {detections_dir}")
        return False

    # Process each JSON file
    for json_file in tqdm(json_files, desc=f"Processing frames for {participant_id}"):
        frame_number = extract_frame_number(json_file.name)
        if frame_number is None:
            print(f"Warning: Invalid frame number in {json_file.name}")
            continue

        total_seconds, time_str = calculate_time(frame_number, fps=fps)

        try:
            with open(json_file, 'r') as f:
                detections = json.load(f)
            if not isinstance(detections, list):
                print(f"Warning: Invalid detections format in {json_file.name}")
                continue
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            continue

        what, where = determine_what_where(detections, priority_order)
        blue_dot_center = get_blue_dot_center(detections)
        
        data.append({
            "Participant": participant_id,
            "Frame Number": frame_number,
            "Time": time_str,
            "What": what,
            "Where": where,
            "Onset": f"{total_seconds:.2f}",
            "Offset": f"{(total_seconds + 1/fps):.2f}",
            "Blue Dot Center": blue_dot_center  # Column H
        })

    if data:
        df = pd.DataFrame(data, columns=[
            "Participant",
            "Frame Number",
            "Time",
            "What",
            "Where",
            "Onset",
            "Offset",
            "Blue Dot Center"  # 8th column
        ])
        df.to_csv(output_csv, index=False)
        print(f"Data sheet saved to: {output_csv}")
        return True
    return False

def find_and_process_all_detections(base_dir):
    """Recursively find and process all 'detections' directories under the base directory."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"Base directory does not exist: {base_dir}")
        return

    # Find all 'detections' directories
    detection_dirs = list(base_dir.rglob("detections"))
    
    if not detection_dirs:
        print(f"No 'detections' directories found under {base_dir}")
        return

    print(f"Found {len(detection_dirs)} 'detections' directories to process.")
    
    processed_count = 0
    failed_dirs = []

    for detections_dir in detection_dirs:
        try:
            # Skip if not under 'inference_output'
            if 'inference_output' not in str(detections_dir):
                continue
                
            print(f"\nProcessing directory: {detections_dir}")
            
            # Get participant directory (two levels up from detections)
            participant_dir = detections_dir.parent.parent
            participant_id = participant_dir.name
            print(f"Participant ID: {participant_id}")

            # Process the detections
            if process_detections(detections_dir, participant_dir, participant_id):
                processed_count += 1
            else:
                failed_dirs.append(str(detections_dir))
                
        except Exception as e:
            print(f"Error processing {detections_dir}: {e}")
            failed_dirs.append(str(detections_dir))

    print("\nProcessing Summary:")
    print(f"Total directories found: {len(detection_dirs)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed to process: {len(failed_dirs)}")
    
    if failed_dirs:
        print("\nFailed directories:")
        for dir_path in failed_dirs:
            print(f"- {dir_path}")

def main():
    """Main function."""
    base_dir = Path("/path/to/base_directory/Adult")  # Replaced private path
    print("Starting detection processing...")
    find_and_process_all_detections(base_dir)
    print("\nAll processing completed.")

if __name__ == "__main__":
    main()
