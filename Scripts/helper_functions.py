"""
Author: og
Date: 2023-08-11
"""
import sys
import os


def change_working_directory():
    """
    Change the working directory to the the currently executing scripts directory on Windows.

    Returns:
        str: Directory of the script.
    """
    script_path = os.path.abspath(sys.argv[0])
    new_working_directory = os.path.dirname(r"{}".format(script_path))
    os.chdir(new_working_directory)
