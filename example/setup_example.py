import os
import glob
import re

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
