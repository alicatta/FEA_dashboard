import pandas as pd
import os

def read_file(filepath):
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file type for: {filepath}")

def check_directory_in_directory(directory, name):
    """
    Checks if a 'plots' directory exists in the given directory. If not, it creates one.
    Returns the path to the 'plots' directory.
    """
    check_directory = os.path.join(directory, name)
    if not os.path.exists(check_directory):
        os.makedirs(check_directory)
    return check_directory