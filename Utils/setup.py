# setup file with utility functions for package installation and setup
import subprocess
import importlib
import sys

def check_and_install_packages(package_list):
    """
    Checks for required packages and installs them if not found.
    This function integrates functionality to operate in Jupyter environments (including 
    Google Colab) or standard IDEs (VSCode, PyCharm, etc.).
    Parameters:
    -----------
    package_list : list
        A list of package names (as strings) to check and install if necessary.
    Returns:
    --------
    None
    """
    for package in package_list:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"{package} not found, installing...")
            try:
                # Check if in a Jupyter (IPython) environment
                if 'get_ipython' in globals():
                    print("Using Jupyter magic command to install.")
                    get_ipython().system(f'pip install {package}')
                else:
                    # Fallback to standard IDE installation method
                    subprocess.run(
                        [sys.executable, '-m', 'pip', 'install', package], 
                        check=True, 
                        capture_output=True
                    )
            except Exception as e:
                print(f"{package} not installed: {e}")
            # Try importing the package again after installation
            try:
                importlib.import_module(package)
            except ImportError as e:
                print(f"Failed to import {package} after installation: {e}")