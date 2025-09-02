
#!/usr/bin/env python3
"""
Installation script for PACT 3.0 dependencies
"""

import subprocess
import sys
import os

def run_command(command, cwd=None):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(command, shell=True, check=True, cwd=cwd, 
                              capture_output=True, text=True)
        print(f"✅ {command}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Installing PACT 3.0 dependencies...")
    
    # Change to PACT3.0 directory
    pact_dir = "PACT3.0"
    if not os.path.exists(pact_dir):
        print(f"❌ Directory {pact_dir} not found!")
        sys.exit(1)
    
    # Install dependencies
    commands = [
        "pip install -r requirements.txt",
        "pip install -e ."
    ]
    
    for cmd in commands:
        if not run_command(cmd, cwd=pact_dir):
            print(f"❌ Failed to run: {cmd}")
            sys.exit(1)
    
    print("✅ All dependencies installed successfully!")
    print("🎉 PACT 3.0 is ready to use!")
    print("📝 Don't forget to set up your environment variables in the Secrets tab")

if __name__ == "__main__":
    main()
