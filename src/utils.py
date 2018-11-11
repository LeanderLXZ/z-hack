import os


def check_dir(path_list):
    """
    Check if directories exit or not.
    """
    for dir_path in path_list:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


def thin_line():
    print('-' * 55)


def thick_line():
    print('=' * 55)
