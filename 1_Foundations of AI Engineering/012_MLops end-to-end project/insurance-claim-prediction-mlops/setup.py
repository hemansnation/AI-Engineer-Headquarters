from setuptools import find_packages, setup
from typing import List


def gather_requirements(filename: str = "requirements.txt") -> List[str]:
    """
    Reads and parses dependencies from a requirements file.

    Args:
        filename (str): Path to the requirements file.

    Returns:
        List[str]: A list of required packages, excluding editable installs.
    """
    try:
        # Open the requirements file and read all lines
        with open(filename, "r") as file:
            return [
                line.strip()  # Remove leading/trailing whitespace
                for line in file.readlines()  # Read each line
                if line.strip() and line.strip() != "-e ."  # Ignore empty lines and '-e .'
            ]
    except FileNotFoundError:
        # Raise a specific error if the file doesn't exist
        raise FileNotFoundError(f"'{filename}' not found.")
    except Exception as e:
        # Catch and raise any other exceptions
        raise RuntimeError(f"An error occurred while reading {filename}: {e}")


# The setup() function defines metadata and configuration for the package
setup(
    name="insurance-claim-prediction",  # Package name
    version="0.0.1",  # Initial version
    author="Tanmay Chakraborty",  # Author name
    author_email="chakrabortytanmay326@gmail.com",  # Contact email
    packages=find_packages(),  # Automatically discover all packages and subpackages
    install_requires=gather_requirements(),  # List of dependencies from requirements.txt
)
