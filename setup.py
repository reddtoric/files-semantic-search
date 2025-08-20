#!/usr/bin/env python3
"""
Setup script for semantic file search tool
Helps with installation and initial configuration
"""

import subprocess
import sys
import os
import platform
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python and try again")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro} (compatible)")
    return True

def check_dependency(package_name, import_name=None):
    """Check if a dependency is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            print(f"✅ {package_name}: Installed")
            return True
        else:
            print(f"❌ {package_name}: Not found")
            return False
    except ImportError:
        print(f"❌ {package_name}: Import error")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n🔧 INSTALLING DEPENDENCIES")
    print("=" * 40)
    
    # Basic dependencies
    basic_deps = [
        "chromadb",
        "sentence-transformers", 
        "unstructured",
        "python-magic-bin",
        "torch",
        "torchvision",
        "torchaudio"
    ]
    
    print("Installing basic dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade"
        ] + basic_deps, check=True)
        print("✅ Basic dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing basic dependencies: {e}")
        return False
    
    return True

def install_gpu_support():
    """Install GPU-enabled PyTorch"""
    print("\n🚀 INSTALLING GPU SUPPORT")
    print("=" * 40)
    
    response = input("Do you want to install GPU acceleration support? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("⏭️ Skipping GPU support installation")
        return True
    
    print("Installing GPU-enabled PyTorch...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ], check=True)
        print("✅ GPU support installed successfully!")
        print("💡 GPU acceleration will be auto-detected when you run the tool")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing GPU support: {e}")
        print("💡 You can still use the tool with CPU-only mode")
        return False
    
    return True

def test_installation():
    """Test if the installation works"""
    print("\n🧪 TESTING INSTALLATION")
    print("=" * 40)
    
    # Test core imports
    test_packages = [
        ("chromadb", "chromadb"),
        ("sentence-transformers", "sentence_transformers"),
        ("unstructured", "unstructured"),
        ("torch", "torch"),
    ]
    
    all_good = True
    for package_name, import_name in test_packages:
        if not check_dependency(package_name, import_name):
            all_good = False
    
    if all_good:
        print("\n✅ All dependencies are working!")
        
        # Test GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🚀 GPU acceleration: Available ({torch.cuda.device_count()} GPU(s))")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    print(f"   GPU {i}: {gpu_name}")
            else:
                print("💻 GPU acceleration: Not available (CPU mode will be used)")
        except:
            print("⚠️ Could not test GPU availability")
        
        return True
    else:
        print("\n❌ Some dependencies are missing or not working")
        return False

def create_example_config():
    """Create example configuration and test files"""
    print("\n📝 CREATING EXAMPLE FILES")
    print("=" * 40)
    
    response = input("Create sample test files for trying out the search tool? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("⏭️ Skipping sample file creation")
        return
    
    # Create a test directory with sample files
    test_dir = "test_files"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"✅ Created {test_dir}/ directory")
        
        # Create sample files for testing (simple, non-sensitive content)
        sample_files = {
            "readme.md": """# Sample Project
This is a sample project for testing the semantic file search tool.

## Features
- Data processing algorithms
- File handling utilities
- Configuration management
- Error handling patterns
""",
            "data_utils.py": """# Data processing utilities
import pandas as pd

def load_data(filename):
    \"\"\"Load data from CSV file\"\"\"
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(data):
    \"\"\"Clean and preprocess data\"\"\"
    # Remove null values
    cleaned_data = data.dropna()
    return cleaned_data
""",
            "file_utils.js": """// File utility functions
const fs = require('fs');

class FileManager {
    constructor() {
        this.supportedFormats = ['.txt', '.json', '.csv'];
    }
    
    async readFile(filepath) {
        try {
            const content = await fs.promises.readFile(filepath, 'utf8');
            return content;
        } catch (error) {
            console.error('File read error:', error);
            return null;
        }
    }
}
""",
            "config.json": """{
    "application": {
        "name": "Sample App",
        "version": "1.0.0",
        "debug": false
    },
    "logging": {
        "level": "INFO",
        "file": "app.log"
    },
    "features": {
        "data_processing": true,
        "file_management": true
    }
}"""
        }
        
        for filename, content in sample_files.items():
            filepath = os.path.join(test_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Created {filepath}")
    else:
        print(f"📁 {test_dir}/ directory already exists")

def run_test_search():
    """Run a test search to verify everything works"""
    print("\n🔍 RUNNING TEST SEARCH")
    print("=" * 40)
    
    if not os.path.exists("file_search.py"):
        print("❌ file_search.py not found. Please ensure all files are in the current directory.")
        return False
    
    response = input("Run a test search on the sample files? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("⏭️ Skipping test search")
        return True
    
    print("\n🚀 Running test search for 'machine learning'...")
    try:
        result = subprocess.run([
            sys.executable, "file_search.py", 
            "machine learning algorithms",
            "-r", "test_files",
            "--model", "A"
        ], capture_output=True, text=True, timeout=60)
        
        print("🔍 Search completed!")
        print("📋 Output:")
        print("-" * 40)
        print(result.stdout)
        if result.stderr:
            print("⚠️ Warnings/Errors:")
            print(result.stderr)
        print("-" * 40)
        
        if result.returncode == 0:
            print("✅ Test search successful!")
            return True
        else:
            print(f"❌ Test search failed with code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test search timed out (this can happen on first run)")
        print("💡 Try running manually: python file_search.py \"machine learning\" -r test_files")
        return False
    except Exception as e:
        print(f"❌ Error running test search: {e}")
        return False

def main():
    """Main setup function"""
    print("🔍 SEMANTIC FILE SEARCH - SETUP")
    print("=" * 50)
    print("This script will help you install and configure the semantic file search tool.")
    print()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check current dependencies
    print("\n📋 CHECKING CURRENT DEPENDENCIES")
    print("=" * 40)
    
    missing_deps = []
    core_deps = [
        ("chromadb", "chromadb"),
        ("sentence-transformers", "sentence_transformers"), 
        ("unstructured", "unstructured"),
        ("torch", "torch"),
    ]
    
    for package_name, import_name in core_deps:
        if not check_dependency(package_name, import_name):
            missing_deps.append(package_name)
    
    # Install missing dependencies
    if missing_deps:
        print(f"\n🔧 Missing dependencies: {', '.join(missing_deps)}")
        response = input("Install missing dependencies? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            if not install_dependencies():
                print("❌ Installation failed. Please check errors above.")
                return 1
        else:
            print("⚠️ Cannot proceed without required dependencies")
            return 1
    else:
        print("\n✅ All core dependencies are already installed!")
    
    # Install GPU support
    install_gpu_support()
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return 1
    
    # Create example files
    create_example_config()
    
    # Run test search
    run_test_search()
    
    # Final instructions
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("✅ Installation successful! You can now use the semantic file search tool.")
    print()
    print("📚 Quick start commands:")
    print('   python file_search.py "your search query"')
    print('   python file_search.py "machine learning" --cache')
    print('   python file_search.py "database code" --model fast')
    print()
    print("📖 For detailed help:")
    print("   python file_search.py --help")
    print("   python examples.py")
    print()
    print("🔧 Performance tips:")
    print("   • Use --cache for faster repeated searches")
    print("   • Use --model fast for large codebases") 
    print("   • Use --debug to see detailed information")
    print()
    print("📁 Test files created in test_files/ directory")
    print("🚀 Ready to search! Happy coding!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
