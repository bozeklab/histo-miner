import os
import sys
import glob
import re
from pathlib import Path
import requests


script_dir = os.path.dirname(os.path.abspath(__file__))



def download_from_zenodo(url: str, output_path: str):
    response = requests.get(url)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        f.write(response.content)

    print(f"Downloaded to {output_path}")

def create_folders():  
    dirs = [
        script_dir + "../../data/",
        script_dir + "../../data/feature_rank/"
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)




if __name__ == "__main__":

    # Create all the needed folders
    create_example_folders()

    # Download tcheckpoints from Zenodo 
    zenodo_links = [,
        "https://zenodo.org/records/15836085/files/Ranking_of_features.json?download=1",
    ]
    file_size = [
        "",
    ]
    output_folder_name = [
        "../../data/feature_rank/",
    ]


    k = 0
    for url in zenodo_links:
        
        output_folder_path = script_dir + output_folder_name[k]
        filename = url.split("/")[-1].split("?")[0]
        output_path = os.path.join(output_folder_path, filename)

        if not os.path.exists(output_path): 
            print("Downloading {} ({})...".format(filename, file_size[k]))
            download_from_zenodo(url, output_path)
        else:
            print("{} already downloaded".format(filename))
       

    print("Feature rank downloaded!")