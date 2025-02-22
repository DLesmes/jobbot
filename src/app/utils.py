"""util functions"""
import json


def save_json(pathfile:str, list_dicts):
    json.dump(list_dicts, open(pathfile, 'w'))
    print(f'storing file at: {pathfile}')


def open_json(pathfile:str):
    print(f'reading file at: {pathfile}')
    loaded_file = json.load(open(pathfile, 'r'))
    return loaded_file
