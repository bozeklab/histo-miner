import os
import json

def remove_areas_keys(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if 'areas' not in k.lower():
                new_dict[k] = remove_areas_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [remove_areas_keys(item) for item in obj]
    else:
        return obj

def try_encodings(file_path, encodings=None):
    """
    Attempt to open and load a JSON file with multiple encodings.
    Returns the loaded JSON data if successful, or None if it fails all attempts.
    """
    if encodings is None:
        encodings = ["utf-8", "latin-1", "cp1252"]
    
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, errors="replace") as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    return None  # Could not decode with given encodings

def process_json_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            data = try_encodings(file_path)
            if data is None:
                print(f"Skipping {filename}: could not read with any known encoding.")
                continue
            
            updated_data = remove_areas_keys(data)

            # Write back in UTF-8
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Replace with the path to your folder of JSON files
    folder_path = '/data/lsancere/Data_General/Collaborators_Dataset/cSCC_CPI_all/Analyses_cSCC_CPI/tissue_analyser_output_allpreprint_35s_noarea/'
    process_json_files(folder_path)
    print("All JSON files have been processed!")
