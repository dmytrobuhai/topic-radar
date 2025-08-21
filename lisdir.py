import os

root_dir = os.path.dirname(os.path.abspath(__file__))

def list_dir(path, prefix=""):
    for item in os.listdir(path):
        if item == ".venv" or item == "__pycache__" or item == ".git":
            continue
        item_path = os.path.join(path, item)
        print(f"{prefix}{item}")
        if os.path.isdir(item_path):
            list_dir(item_path, prefix + "    ")

list_dir(root_dir)