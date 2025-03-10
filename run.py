#!/usr/bin/env python3
import os
import sys
import subprocess
import platform

# Check Python version
python_version = tuple(map(int, platform.python_version_tuple()))
if python_version < (3, 10):
    print("Error: TinyGrad requires Python 3.10 or newer")
    print(f"Current Python version: {platform.python_version()}")
    print("Please install Python 3.10+ and try again")
    
    # Try to find Python 3.10+ on the system
    for version in ["3.10", "3.11", "3.12"]:
        python_path = f"/usr/local/bin/python{version}"
        if os.path.exists(python_path):
            print(f"\nFound Python {version} at {python_path}")
            print(f"Try running: {python_path} {' '.join(sys.argv)}")
            sys.exit(1)
    sys.exit(1)

# Add the TinyGrad directory to the Python path
tinygrad_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinygrad")
os.environ["PYTHONPATH"] = tinygrad_path + os.pathsep + os.environ.get("PYTHONPATH", "")

# Run the specified script with the correct Python path
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <script_to_run.py> [args...]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    script_args = sys.argv[2:]
    
    cmd = [sys.executable, script_path] + script_args
    subprocess.run(cmd)