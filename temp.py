#!/usr/bin/env python3

import os
import numpy as np
from io import StringIO

# Hardcode the path to the folder containing your CSV files.
FOLDER_PATH = "/data/lsancere/Data_General/Other_Dataset/histo-miner/\
validation_tables/boxplots/pythonready/csv/tumor/"


def convert_csv_to_npy(csv_filepath):
    """
    Reads a CSV file and saves it as a NPY file with the same base name.

    :param csv_filepath: The path to the CSV file to be converted.
    """
    # Load the CSV data (adjust parameters if needed, e.g., delimiter, skiprows, etc.)
    data = np.loadtxt(csv_filepath, delimiter=',')
    
    # Define the output filename by replacing .csv with .npy
    npy_filepath = csv_filepath.rsplit('.', 1)[0] + '.npy'
    
    # Save the data as a NPY file
    np.save(npy_filepath, data)
    print(f"Converted: {csv_filepath} -> {npy_filepath}")

def convert_all_csv_in_folder(folder_path):
    """
    Recursively walks through the folder and subfolders to find CSV files
    and convert them to NPY, skipping files that start with '.'.
    
    :param folder_path: The path to the folder where CSV files are located.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Skip files that start with a dot
            if file.startswith('.'):
                continue
            
            # Process only files that have a CSV extension
            if file.lower().endswith('.csv'):
                csv_filepath = os.path.join(root, file)
                convert_csv_to_npy(csv_filepath)

def main():
    # Make sure the hardcoded path is valid
    if not os.path.isdir(FOLDER_PATH):
        print(f"Error: The provided path '{FOLDER_PATH}' is not a directory.")
        return
    
    convert_all_csv_in_folder(FOLDER_PATH)

if __name__ == '__main__':
    main()











# def convert_csv_to_npy(csv_filepath):
#     """
#     Reads a CSV file (with potential non-UTF-8 characters) and saves it as a NPY file.
#     """
#     # Choose an encoding that matches your file best, e.g., 'utf-8', 'latin1', or 'cp1252'.
#     chosen_encoding = 'utf-8'
    
#     # 1) Read the file in binary mode
#     with open(csv_filepath, 'rb') as f:
#         raw_data = f.read()
    
#     # 2) Decode the raw bytes, replacing invalid characters with � (replacement char)
#     decoded_data = raw_data.decode(chosen_encoding, errors='replace')
    
#     # 3) Split the decoded text into individual lines
#     lines = decoded_data.splitlines()
    
#     # 4) Use StringIO to mimic a file-like object for np.loadtxt()
#     data = np.loadtxt(StringIO('\n'.join(lines)), delimiter=',')
    
#     # 5) Build the NPY path and save
#     npy_filepath = csv_filepath.rsplit('.', 1)[0] + '.npy'
#     np.save(npy_filepath, data)
#     print(f"Converted: {csv_filepath} -> {npy_filepath}")

# def convert_all_csv_in_folder(folder_path):
#     """
#     Recursively walks through the folder and subfolders to find CSV files
#     and convert them to NPY.
#     """
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith('.csv'):
#                 csv_filepath = os.path.join(root, file)
#                 convert_csv_to_npy(csv_filepath)

# def main():
#     if not os.path.isdir(FOLDER_PATH):
#         print(f"Error: The provided path '{FOLDER_PATH}' is not a directory.")
#         return
    
#     convert_all_csv_in_folder(FOLDER_PATH)

# if __name__ == '__main__':
#     main()










# import os
# import json

# def remove_areas_keys(obj):
#     if isinstance(obj, dict):
#         new_dict = {}
#         for k, v in obj.items():
#             if 'areas' not in k.lower():
#                 new_dict[k] = remove_areas_keys(v)
#         return new_dict
#     elif isinstance(obj, list):
#         return [remove_areas_keys(item) for item in obj]
#     else:
#         return obj

# def try_encodings(file_path, encodings=None):
#     """
#     Attempt to open and load a JSON file with multiple encodings.
#     Returns the loaded JSON data if successful, or None if it fails all attempts.
#     """
#     if encodings is None:
#         encodings = ["utf-8", "latin-1", "cp1252"]
    
#     for enc in encodings:
#         try:
#             with open(file_path, "r", encoding=enc, errors="replace") as f:
#                 return json.load(f)
#         except (UnicodeDecodeError, json.JSONDecodeError):
#             pass
#     return None  # Could not decode with given encodings

# def process_json_files(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".json"):
#             file_path = os.path.join(folder_path, filename)
            
#             data = try_encodings(file_path)
#             if data is None:
#                 print(f"Skipping {filename}: could not read with any known encoding.")
#                 continue
            
#             updated_data = remove_areas_keys(data)

#             # Write back in UTF-8
#             with open(file_path, "w", encoding="utf-8") as f:
#                 json.dump(updated_data, f, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     # Replace with the path to your folder of JSON files
#     folder_path = '/data/lsancere/Data_General/Collaborators_Dataset/cSCC_CPI_all/Analyses_cSCC_CPI/tissue_analyser_output_allpreprint_35s_noarea/'
#     process_json_files(folder_path)
#     print("All JSON files have been processed!")
