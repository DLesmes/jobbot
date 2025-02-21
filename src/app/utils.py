"""util functions"""
#base
import os
import json


def save_json(pathfile:str, list_dicts):
    json.dump(list_dicts, open(pathfile, 'w'))
    print(f'storing file at: {pathfile}')


def open_json(pathfile:str):
    print(f'reading file at: {pathfile}')
    loaded_file = json.load(open(pathfile, 'r'))
    return loaded_file

def get_file_paths(directory):
    # List to hold paths of all files
    file_paths = []

    # Walk through directory
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths
