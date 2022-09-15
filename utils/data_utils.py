import json


def read_file(path):
    """
    Reads data from jsom file.
    :param path: path to read json file from
    :return: data
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def write_file(path, file):
    """
    Writes data to json file.
    :param path: path to write json to
    :param file: data to write
    """
    with open(path, 'w') as outfile:
        json.dump(file, outfile)
