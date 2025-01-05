import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Dictionaries for parsing spelled-out numbers.
UNITS = {
    "Zero": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
    "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10,
    "Eleven": 11, "Twelve": 12, "Thirteen": 13, "Fourteen": 14,
    "Fifteen": 15, "Sixteen": 16, "Seventeen": 17, "Eighteen": 18,
    "Nineteen": 19
}
TENS = {
    "Twenty": 20, "Thirty": 30, "Forty": 40, "Fourty": 40,
    "Fifty": 50, "Sixty": 60, "Seventy": 70, "Eighty": 80,
    "Ninety": 90
}

def parse_spelled_number(spelled_str: str) -> int:
    """
    Splits a spelled-out number (like 'FiftySix') into parts
    and sums them. E.g., 'FiftySix' -> 56, 'FortyOne' -> 41.
    """
    parts = re.findall(r'[A-Z][a-z]+', spelled_str)  # Break on capital letters
    total = 0
    for p in parts:
        if p in TENS:
            total += TENS[p]
        elif p in UNITS:
            total += UNITS[p]
    return total

def get_participant_type_and_age_months(participant_folder: Path) -> (str, int):
    """
    Determines if folder belongs to an adult or an infant
    by checking the path. Parses spelled-out number in the
    folder name to figure out age. For adults, convert years to months.
    For infants, the spelled number is already in months.
    """
    folder_str = str(participant_folder).lower()
    folder_name = participant_folder.name
    first_chunk = folder_name.split("-")[0]
    spelled_number_value = parse_spelled_number(first_chunk)

    if "adult" in folder_str:
        return "adult", spelled_number_value * 12
    elif "infant" in folder_str:
        return "infant", spelled_number_value
    else:
        return "unknown", 0

def get_latest_datasheet_dir(parent_dir: Path):
    """
    Find the 'datasheet' directory OR the subdirectory with
    the largest numeric suffix if it starts with 'datasheet-'.
    If there's a plain 'datasheet' folder, return it immediately.
    Otherwise, scan for datasheet-* directories.
    Returns the Path to that directory, or None if none found.
    """
    datasheet_dir = parent_dir / "datasheet"
    if datasheet_dir.is_dir():
        return datasheet_dir

    run_dirs = []
    for child in parent_dir.iterdir():
        if child.is_dir() and child.name.startswith("datasheet-"):
            parts = child.name.split("datasheet-")
            if len(parts) < 2:
                continue
            suffix_str = parts[-1].replace("-", "")
            try:
                suffix_num = int(suffix_str)
                run_dirs.append((suffix_num, child))
            except ValueError:
                pass
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda x: x[0])
    return run_dirs[-1][1]

def get_latest_classification_dir(parent_dir: Path):
    """
    Find the classification-run-* directory with the largest numeric suffix.
    Returns Path or None if none found.
    """
    run_dirs = []
    for child in parent_dir.iterdir():
        if child.is_dir() and child.name.startswith("classification-run-"):
            parts = child.name.split("classification-run-")
            if len(parts) < 2:
                continue
            suffix_str = parts[-1].replace("-", "")
            try:
                suffix_num = int(suffix_str)
                run_dirs.append((suffix_num, child))
            except ValueError:
                pass
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda x: x[0])
    return run_dirs[-1][1]

def consolidate_data(datasheet_csv: Path, detections_csv: Path, output_path: Path,
                     participant_type: str, age_months: int):
    """
    Merges datasheet-[participant id].csv (which has the new column H)
    with detections_summary_corrected.csv on frame number.
    Keeps only rows from datasheet that match frames in detections_csv.
    Columns from datasheet appear on the left, columns from detections on the right.
    Adds columns for participant_type and participant_age_months.
    Also adds participant_age_years with one decimal place.
    """
    df_data_sheet = pd.read_csv(datasheet_csv)
    df_detections = pd.read_csv(detections_csv)

    # Merge on Frame Number
    df_merged = pd.merge(
        df_data_sheet,
        df_detections,
        left_on="Frame Number",
        right_on="frame_number",
        how="inner"
    )

    # Remove the extra column (frame_number) if it exists
    if "frame_number" in df_merged.columns:
        df_merged.drop(columns=["frame_number"], inplace=True)

    df_merged["participant_type"] = participant_type
    df_merged["participant_age_months"] = age_months
    
    # Create a new column in years, rounded to 1 decimal place
    df_merged["participant_age_years"] = (df_merged["participant_age_months"] / 12).round(1)

    # Write to CSV
    try:
        df_merged.to_csv(output_path, index=False)
    except PermissionError:
        print(f"Permission denied: {output_path}\nSkipping this participant.")
        return

def main():
    root_dir = Path("/path/to/base_directory/Adult")
    if not root_dir.exists():
        print(f"Root directory does not exist: {root_dir}")
        return

    print(f"Searching in: {root_dir}\n")

    candidate_parents = set()
    for subdir in root_dir.rglob("*"):
        if not subdir.is_dir():
            continue
        if (subdir.name == "datasheet"
            or subdir.name.startswith("datasheet-")
            or subdir.name.startswith("classification-run-")):
            candidate_parents.add(subdir.parent)

    if not candidate_parents:
        print("No matching datasheet / classification-run-* directories found.")
        return

    for parent in tqdm(candidate_parents, desc="Processing participant directories"):
        ds_dir = get_latest_datasheet_dir(parent)
        cls_dir = get_latest_classification_dir(parent)

        if ds_dir is None or cls_dir is None:
            continue

        # Find the single CSV in the datasheet directory (handles "datasheet-[participant].csv")
        data_sheet_csv_candidates = list(ds_dir.glob("*.csv"))
        if not data_sheet_csv_candidates:
            continue
        data_sheet_csv = data_sheet_csv_candidates[0]

        detections_csv = cls_dir / "detections_summary_corrected.csv"
        if not data_sheet_csv.exists() or not detections_csv.exists():
            continue

        participant_folder = parent
        participant_name = participant_folder.name
        # Remove trailing (year) if present
        participant_name_no_year = re.sub(r"\s*\(\d{4}\)$", "", participant_name)

        # Determine participant type and age
        p_type, p_age_months = get_participant_type_and_age_months(participant_folder)

        output_csv = participant_folder / f"{participant_name_no_year}.csv"

        # Skip if output file already exists
        if output_csv.exists():
            print(f"Output file already exists, skipping: {output_csv}")
            continue

        print(f"\nConsolidating data for: {participant_folder}")
        print(f" - Using datasheet dir: {ds_dir}")
        print(f" - Using classification dir: {cls_dir}")
        print(f" - Output: {output_csv}")

        consolidate_data(data_sheet_csv, detections_csv, output_csv, p_type, p_age_months)

    print("\nConsolidation complete. One output per participant directory.\n")

if __name__ == "__main__":
    main()
