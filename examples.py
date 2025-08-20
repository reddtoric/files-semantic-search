#!/usr/bin/env python3
"""
Example usage scripts for semantic file search
Run this to see various usage patterns and examples
"""

import subprocess
import sys
import os

def run_example(description, command, explain=None):
    """Run an example command with description"""
    print(f"\n{'='*60}")
    print(f"📚 EXAMPLE: {description}")
    print(f"{'='*60}")
    print(f"Command: python {command}")
    if explain:
        print(f"Explanation: {explain}")
    print(f"{'='*60}")
    
    # Ask user if they want to run it
    response = input("🔍 Run this example? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        try:
            # Run the command
            full_command = f"python {command}"
            print(f"\n🚀 Running: {full_command}")
            print("-" * 40)
            subprocess.run(full_command, shell=True, check=False)
            print("-" * 40)
            print("✅ Example completed!")
        except KeyboardInterrupt:
            print("\n⚠️ Example interrupted by user")
        except Exception as e:
            print(f"❌ Error running example: {e}")
    else:
        print("⏭️ Skipped")


def main():
    """Run example scenarios"""
    print("🔍 SEMANTIC FILE SEARCH - INTERACTIVE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates various ways to use the semantic file search tool.")
    print("Each example shows a different use case with explanations.")
    print("\nNote: Examples use the current directory. Change paths as needed.")
    print("=" * 60)
    
    # Check if main script exists
    if not os.path.exists("file_search.py"):
        print("❌ Error: file_search.py not found in current directory")
        print("Please run this script from the directory containing file_search.py")
        return 1
    
    examples = [
        {
            "description": "Basic Search - Find machine learning related files",
            "command": 'file_search.py "machine learning algorithms"',
            "explain": "Simple search using default settings. Good for getting started."
        },
        {
            "description": "Fast Search - Quick results for large codebases", 
            "command": 'file_search.py "database connection" --model fast --cache',
            "explain": "Uses the fastest AI model and enables caching for speed."
        },
        {
            "description": "High Quality Search - Best accuracy for important searches",
            "command": 'file_search.py "authentication logic" --model best -k 5 -s 0.7',
            "explain": "Uses the most accurate model, limits to 5 results, higher similarity threshold."
        },
        {
            "description": "Code-Specific Search - Optimized for programming files",
            "command": 'file_search.py "error handling patterns" --model code --extensions .py .js .ts',
            "explain": "Uses code-optimized model and searches only programming files."
        },
        {
            "description": "Debug Mode - See detailed information about the search",
            "command": 'file_search.py "utility functions" --debug --min-score 0.0',
            "explain": "Shows all results with scores and detailed processing information."
        },
        {
            "description": "Configuration Search - Find config and setup files",
            "command": 'file_search.py "database settings" --extensions .json .yaml .cfg .env',
            "explain": "Searches only configuration file types for database-related settings."
        },
        {
            "description": "Documentation Search - Find README, docs, and guides",
            "command": 'file_search.py "installation instructions" --extensions .md .txt .rst',
            "explain": "Searches documentation files for installation information."
        },
        {
            "description": "Performance Optimized - For very large projects",
            "command": 'file_search.py "API endpoints" --model fast --threads 16 --max-files 2000 --cache',
            "explain": "Optimized for speed: fast model, many threads, file limit, caching enabled."
        },
        {
            "description": "Multi-language Search - For international projects",
            "command": 'file_search.py "user interface" --model multi',
            "explain": "Uses multilingual model that understands multiple languages."
        },
        {
            "description": "Performance Analysis - See where time is spent",
            "command": 'file_search.py "test files" --perf-report --debug --cache',
            "explain": "Shows detailed performance breakdown and debug information."
        }
    ]
    
    print(f"\n📋 Available Examples ({len(examples)} total):")
    for i, example in enumerate(examples, 1):
        print(f"  {i:2d}. {example['description']}")
    
    print(f"\n🎯 Choose how to proceed:")
    print("  1. Run all examples (recommended for learning)")
    print("  2. Choose specific examples") 
    print("  3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Run all examples
        print("\n🚀 Running all examples...")
        for i, example in enumerate(examples, 1):
            print(f"\n📍 Example {i} of {len(examples)}")
            run_example(example["description"], example["command"], example["explain"])
            
            if i < len(examples):
                input("\n⏸️  Press Enter to continue to next example...")
    
    elif choice == "2":
        # Choose specific examples
        while True:
            try:
                selection = input(f"\nEnter example number (1-{len(examples)}) or 'q' to quit: ").strip()
                if selection.lower() == 'q':
                    break
                    
                example_num = int(selection)
                if 1 <= example_num <= len(examples):
                    example = examples[example_num - 1]
                    run_example(example["description"], example["command"], example["explain"])
                else:
                    print(f"❌ Please enter a number between 1 and {len(examples)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user")
                break
    
    elif choice == "3":
        print("👋 Goodbye!")
        return 0
    
    else:
        print("❌ Invalid choice. Please run the script again.")
        return 1
    
    print("\n🎉 Examples session completed!")
    print("\n💡 Next steps:")
    print("  • Try your own search queries")
    print("  • Enable caching with --cache for faster repeated searches")
    print("  • Use --debug to understand how the tool works")
    print("  • Read README.md for comprehensive documentation")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
