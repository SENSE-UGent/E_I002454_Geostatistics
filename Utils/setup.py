import subprocess
import importlib
import sys
import pkg_resources
from packaging import version

def check_and_install_packages(package_list):
    """
    Checks for required packages and installs them if not found or if version requirements aren't met.
    This function integrates functionality to operate in Jupyter environments (including 
    Google Colab) or standard IDEs (VSCode, PyCharm, etc.).
    
    Parameters:
    -----------
    package_list : list
        A list of package names or package specifications with versions.
        Examples:
        - ['numpy', 'pandas']  # Install latest versions
        - ['numpy>=1.20.0', 'pandas==1.3.0']  # With version constraints
        - ['scikit-learn~=1.0.0']  # Compatible version
    
    Returns:
    --------
    None
    """
    for package_spec in package_list:
        # Parse package name and version requirement
        package_name, version_req = _parse_package_spec(package_spec)
        
        try:
            # Check if package is installed
            installed_package = pkg_resources.get_distribution(package_name)
            installed_version = installed_package.version
            
            # Check version compatibility if version requirement is specified
            if version_req and not _is_version_compatible(installed_version, version_req):
                print(f"{package_name} {installed_version} found, but {package_spec} is required. Upgrading...")
                _install_package(package_spec)
            else:
                print(f"{package_name} {installed_version} is already installed and compatible.")
                
        except pkg_resources.DistributionNotFound:
            print(f"{package_name} not found, installing {package_spec}...")
            _install_package(package_spec)
        
        # Verify installation
        _verify_installation(package_name, version_req)

def _parse_package_spec(package_spec):
    """Parse package specification to extract name and version requirement."""
    import re
    
    # Match patterns like: numpy>=1.20.0, pandas==1.3.0, scikit-learn~=1.0.0
    match = re.match(r'^([a-zA-Z0-9_-]+)([<>=~!]+.*)?$', package_spec.strip())
    
    if match:
        package_name = match.group(1)
        version_req = match.group(2) if match.group(2) else None
        return package_name, version_req
    else:
        return package_spec.strip(), None

def _is_version_compatible(installed_version, version_req):
    """Check if installed version meets the requirement."""
    try:
        # Create a dummy requirement to use pkg_resources for version checking
        requirement = pkg_resources.Requirement.parse(f"dummy{version_req}")
        return installed_version in requirement
    except Exception:
        return False

def _install_package(package_spec):
    """Install the specified package."""
    try:
        # Check if in a Jupyter (IPython) environment
        if 'get_ipython' in globals():
            print("Using Jupyter magic command to install.")
            get_ipython().system(f'pip install "{package_spec}"')
        else:
            # Fallback to standard IDE installation method
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package_spec], 
                check=True, 
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(f"Installation output: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_spec}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error installing {package_spec}: {e}")

def _verify_installation(package_name, version_req):
    """Verify that the package was installed correctly."""
    try:
        installed_package = pkg_resources.get_distribution(package_name)
        installed_version = installed_package.version
        
        if version_req and not _is_version_compatible(installed_version, version_req):
            print(f"Warning: {package_name} {installed_version} was installed, but may not meet requirement {version_req}")
        else:
            print(f"Successfully verified {package_name} {installed_version}")
            
    except pkg_resources.DistributionNotFound:
        print(f"Warning: {package_name} installation could not be verified")
    except Exception as e:
        print(f"Error verifying {package_name}: {e}")
