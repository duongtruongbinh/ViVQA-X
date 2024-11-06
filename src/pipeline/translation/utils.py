import json
import subprocess
from typing import List


def uninstall_packages(packages: List[str]):
    """
    Uninstall specified packages using pip.

    Args:
        packages (List[str]): A list of package names to uninstall.

    Returns:
        None
    """
    for package in packages:
        try:
            subprocess.check_call(
                ["pip", "uninstall", "-y", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Uninstalled {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error uninstalling {package}: {str(e)}")


def install_requirements(packages: List[str]):
    """
    Install required packages if not already installed.

    Args:
        packages (List[str]): A list of package names to install.

    Returns:
        None
    """
    package_names = [package.split("===")[0] for package in packages]
    uninstall_packages(package_names)

    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call(
                ["pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Installed {package} successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")


def load_json(file_path: str) -> dict:
    """
    Load a JSON file and return its content as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, file_path: str):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The data to save.
        file_path (str): The path to the JSON file.

    Returns:
        None
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
