import os
import glob

def replace_in_file(path, old, new):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    if old in text:
        text = text.replace(old, new)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Reverted: {path}")

def batch_replace(folder, old, new, extensions=('.py', '.sh')):
    for ext in extensions:
        pattern = os.path.join(folder, f'**/*{ext}')
        # recursive glob
        for filepath in glob.glob(pattern, recursive=True):
            replace_in_file(filepath, old, new)

if __name__ == "__main__":
    target_folder = "../scripts/"
    batch_replace(
        folder=target_folder,
        old="/example/example-configs/",
        new="/configs/"
    )

    print("Example unset. You can now use histo-miner from the main folder.")
