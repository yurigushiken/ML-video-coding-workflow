import os
import pandas as pd

def fix_main_csv(main_csv_path):
    """
    Reads the main CSV and adjusts rows where:
      events_corrected in ['ugwo', 'gwo']
      What == 'toy'
      Where == 'other'
    Those rows get 'What' -> 'man' and 'Where' -> 'hands'.
    Returns a list of frame numbers that were changed (changed_frames).
    Prints out which lines changed or if no changes were needed.
    """
    try:
        df = pd.read_csv(main_csv_path)
    except Exception as e:
        print(f"Could not read {main_csv_path} due to error: {e}")
        return []

    required_cols = {'events_corrected', 'What', 'Where', 'Frame Number'}
    if not required_cols.issubset(df.columns):
        print(f"Skipping {main_csv_path}: missing required columns.")
        return []

    condition = (
        df['events_corrected'].isin(['ugwo', 'gwo']) &
        (df['What'] == 'toy') &
        (df['Where'] == 'other')
    )
    changed_indices = df.index[condition].tolist()

    if not changed_indices:
        print(f"No changes needed for {main_csv_path}")
        return []

    print(f"Changes in {main_csv_path}:")
    for idx in changed_indices:
        frame_val = df.at[idx, 'Frame Number']
        old_what = df.at[idx, 'What']
        old_where = df.at[idx, 'Where']
        print(f"  Changing line index {idx}, Frame Number {frame_val} "
              f"from ({old_what}, {old_where}) to (man, hands)")

    df.loc[condition, ['What','Where']] = ['man','hands']
    df.to_csv(main_csv_path, index=False)

    changed_frames = df.loc[changed_indices, 'Frame Number'].tolist()
    return changed_frames

def fix_datasheet_csv(datasheet_path, changed_frames):
    """
    For the datasheet CSV, we adjust the same frames that changed in the main CSV.
    We do NOT check 'events_corrected' since it may not exist.
    Instead, we simply look for rows where:
      'Frame Number' is in changed_frames
      'What' == 'toy'
      'Where' == 'other'
    Then we replace 'What' -> 'man' and 'Where' -> 'hands'.
    Prints out which lines changed or if no changes were needed.
    """
    if not changed_frames:
        return

    try:
        df = pd.read_csv(datasheet_path)
    except Exception as e:
        print(f"Could not read {datasheet_path} due to error: {e}")
        return

    required_cols = {'What', 'Where', 'Frame Number'}
    if not required_cols.issubset(df.columns):
        print(f"Skipping {datasheet_path}: missing required columns.")
        return

    condition = (
        df['Frame Number'].isin(changed_frames) &
        (df['What'] == 'toy') &
        (df['Where'] == 'other')
    )
    changed_indices = df.index[condition].tolist()

    if not changed_indices:
        print(f"No changes needed for {datasheet_path}")
        return

    print(f"Changes in {datasheet_path}:")
    for idx in changed_indices:
        frame_val = df.at[idx, 'Frame Number']
        old_what = df.at[idx, 'What']
        old_where = df.at[idx, 'Where']
        print(f"  Changing line index {idx}, Frame Number {frame_val} "
              f"from ({old_what}, {old_where}) to (man, hands)")

    df.loc[condition, ['What','Where']] = ['man','hands']
    df.to_csv(datasheet_path, index=False)

def process_participant_dir(participant_dir):
    """
    1) Find any main CSV(s) in participant_dir that are not datasheet CSVs.
    2) For each main CSV:
       - Call fix_main_csv and get changed_frames.
    3) In the 'datasheet' subdirectory (if present):
       - Call fix_datasheet_csv with the same changed_frames.
    """
    main_csv_files = []
    for item in os.listdir(participant_dir):
        full_path = os.path.join(participant_dir, item)
        if os.path.isfile(full_path) and item.lower().endswith(".csv"):
            if 'datasheet' not in item.lower():
                main_csv_files.append(full_path)

    for main_csv_path in main_csv_files:
        changed_frames = fix_main_csv(main_csv_path)

        datasheet_dir = os.path.join(participant_dir, "datasheet")
        if os.path.isdir(datasheet_dir):
            for ds_item in os.listdir(datasheet_dir):
                ds_path = os.path.join(datasheet_dir, ds_item)
                if os.path.isfile(ds_path) and ds_item.lower().endswith(".csv"):
                    fix_datasheet_csv(ds_path, changed_frames)

def fix_toy_other_in_gwo_ugwo(root_dir):
    """
    Recursively searches for participant directories under root_dir.
    A participant directory is one where:
     - There's at least one CSV that isn't obviously a datasheet
       or there's a 'datasheet' subdir.
    """
    for root, dirs, files in os.walk(root_dir):
        main_csv_found = any(
            f.lower().endswith(".csv") and ('datasheet' not in f.lower())
            for f in files
        )
        datasheet_subdir_found = any(d.lower() == "datasheet" for d in dirs)

        if main_csv_found or datasheet_subdir_found:
            process_participant_dir(root)

if __name__ == "__main__":
    root_directory = r"/path/to/base_directory"
    fix_toy_other_in_gwo_ugwo(root_directory)
