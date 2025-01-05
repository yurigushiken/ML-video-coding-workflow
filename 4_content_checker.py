import os

def count_files_in_dir(path):
    """
    Returns the number of files (not subdirectories) in the given path.
    If the path doesn't exist, returns None.
    """
    if not os.path.isdir(path):
        return None
    return sum(
        len(files)
        for _, _, files in os.walk(path)
    )

def main():
    base_dir = r"/path/to/base_directory"

    # We'll walk through all directories under base_dir.
    for root, dirs, files in os.walk(base_dir):
        # Check if this directory contains the subfolders we need
        path_1024 = os.path.join(root, "1024x1024_frames")
        path_classif_detections = os.path.join(root, "classification-run-01", "detections")
        path_infer_detections = os.path.join(root, "inference_output", "detections")
        path_infer_visuals = os.path.join(root, "inference_output", "visual_outputs")

        if (os.path.isdir(path_1024) and 
            os.path.isdir(os.path.join(root, "classification-run-01")) and
            os.path.isdir(os.path.join(root, "inference_output"))):
            
            # We have a candidate "participant" folder
            count_1024 = count_files_in_dir(path_1024)
            count_classif_det = count_files_in_dir(path_classif_detections)
            count_infer_det = count_files_in_dir(path_infer_detections)
            count_infer_vis = count_files_in_dir(path_infer_visuals)

            # Print out the results
            print(f"\nParticipant folder: {root}")
            print(f"  1024x1024_frames:             {count_1024}")
            print(f"  classification-run-01/detections: {count_classif_det}")
            print(f"  inference_output/detections:     {count_infer_det}")
            print(f"  inference_output/visual_outputs: {count_infer_vis}")

            counts = [count_1024, count_classif_det, count_infer_det, count_infer_vis]

            # Check if all counts are not None and if they match
            if None not in counts and len(set(counts)) == 1:
                print("  => All counts match.")
            else:
                print("  => Counts do not match or some directories are missing.")
                print("     Detailed counts:")
                labels = ["1024_frames", "classif_det", "infer_det", "infer_vis"]
                for label, c in zip(labels, counts):
                    print("       ", label, "=", c)

if __name__ == "__main__":
    main()
