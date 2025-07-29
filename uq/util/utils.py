import os
import sys
import subprocess

from datetime import datetime


def add_timestamp_to_filename(filename, timestamp=None):
    """
    Add timestamp to filename before the extension.
    
    Args:
        filename (str): Original filename
        timestamp (str, optional): Custom timestamp string. If None, uses current datetime.
    
    Returns:
        str: Filename with timestamp
    
    Example:
        add_timestamp_to_filename("results.hdf5") -> "results_20250718_143025.hdf5"
    """
    if timestamp is None:
        # Different timestamp formats:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        # 20250718_143025
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # 2025-07-18_14-30-25
        # timestamp = datetime.now().strftime("%Y%m%d")               # 20250718 (date only)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M")          # 20250718_1430 (no seconds)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"

def get_festim_python():
    """Get the correct Python executable for FESTIM environment."""
    
    # Method 1: Check if specific conda environment exists
    conda_python = "/home/yehor/miniconda3/envs/festim-env/bin/python3"
    if os.path.exists(conda_python):
        return conda_python
    
    # Method 2: Try to find conda environment dynamically
    try:
        result = subprocess.run(['conda', 'info', '--envs'], 
                              capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if 'festim-env' in line:
                env_path = line.split()[-1]
                python_path = os.path.join(env_path, 'bin', 'python')
                if os.path.exists(python_path):
                    return python_path
    except:
        pass
    
    # Method 3: Fallback to current Python
    print("Warning: FESTIM environment not found, using current Python")
    return sys.executable

def validate_execution_setup():
    """Validate that the execution environment is properly configured."""
    runnable_script = "festim_model_run.py"
    script_path = os.path.join(os.getcwd(), runnable_script)
    
    # Check script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    # Check script is executable
    if not os.access(script_path, os.X_OK):
        print(f"Making script executable: {script_path}")
        os.chmod(script_path, 0o755)
    
    # Check Python executable
    python_exe = get_festim_python()
    if not os.path.exists(python_exe):
        raise FileNotFoundError(f"Python executable not found: {python_exe}")
    
    print(f"✓ Script validation passed: {script_path}")
    print(f"✓ Python executable: {python_exe}")
    return python_exe, script_path
