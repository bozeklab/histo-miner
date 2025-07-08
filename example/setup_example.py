import os
import sys
import glob
import re
from pathlib import Path
import requests


script_dir = os.path.dirname(os.path.abspath(__file__))



def get_parent_path():
    """Get current directory path minus the last folder."""
    return os.path.dirname(os.getcwd())

def replace_in_file(path, old, new):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Use regular expression to replace '/configs/' not followed by '_'
    pattern = re.compile(re.escape(old) + r'(?!_)')
    new_text = pattern.sub(new, text)
    if text != new_text:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_text)
        print(f"Updated: {path}")

def batch_replace(folder, old, new, extensions=('.py', '.sh')):
    for ext in extensions:
        pattern = os.path.join(folder, f'**/*{ext}')
        for filepath in glob.glob(pattern, recursive=True):
            replace_in_file(filepath, old, new)

def process_yaml_folder_recursive(folder_path):
    parent_path = get_parent_path()
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(('.yml', '.yaml')):
                file_path = os.path.join(root, filename)
                replace_in_file(file_path, "/user_histominer_path/", parent_path + "/")



def create_example_folders():  
    dirs = [
        script_dir + "/checkpoints/",
        script_dir + "/data/",
        script_dir + "/data/downsampling/",
        script_dir + "/results/",
        script_dir + "/results/scchovernet_output/",
        script_dir + "/results/sccsegmenter_output/",
        script_dir + "/results/post-processing/",
        script_dir + "/results/tissue_analyser/",
        script_dir + "/cache/"
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)



def download_from_zenodo(url: str, output_path: str):
    response = requests.get(url)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        f.write(response.content)

    print(f"Downloaded to {output_path}")




if __name__ == "__main__":
    # Folder containing .py and .sh scripts to update "/configs/" path
    target_script_folder = "../scripts/"
    batch_replace(
        folder=target_script_folder,
        old="/configs/",
        new="/example/example-configs/"
    )

    # Folder containing config files to replace /user_histominer_path/
    target_config_folder = "./example-configs"
    process_yaml_folder_recursive(target_config_folder)

    print(f'Updated configs to run example.')

    # Create all the paths
    create_example_folders()


    # Download the slide and the checkpoints from Zenodo 
    zenodo_links = [
        "https://zenodo.org/records/15836085/files/wsi_example.ndpi?download=1",
        "https://zenodo.org/records/13970198/files/scchovernet_bestweights.tar?download=1",
        "https://zenodo.org/records/13970198/files/sccsegmenter_bestweights.pth?download=1",
    ]
    file_size = [
        "1.47",
        "0.45",
        "2.7",
    ]
    output_folder_name = [
        "/data/",
        "/checkpoints/",
        "/checkpoints/",
    ]


    k = 0
    for url in zenodo_links:
        
        output_folder_path = script_dir + output_folder_name[k]
        filename = url.split("/")[-1].split("?")[0]
        output_path = os.path.join(output_folder_path, filename)

        if not os.path.exists(output_path): 
            print("Downloading {} ({} GB)...".format(filename, file_size[k]))
            download_from_zenodo(url, output_path)
        else:
            print("{} already downloaded".format(filename))
        
        k += 1

    print("Example setup complete!")
