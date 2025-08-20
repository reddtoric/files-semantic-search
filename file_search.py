#!/usr/bin/env python3
"""
Enhanced semantic file search script
Searches files using vector embeddings and semantic similarity
Now with performance optimizations for faster scanning and processing

USAGE:
    python file_search.py "search query"
    python file_search.py "machine learning code" --model fast --cache
    python file_search.py "authentication logic" -k 5 -s 0.7 --debug

FILES STRUCTURE:
    file_search.py      - Main entry point (this file)
    config.py          - Configuration management
    file_processor.py  - File scanning and text extraction
    vector_db.py       - Vector database and AI operations
    utils.py           - Utilities (spinner, progress, caching)
"""

import os
import sys
import time
import signal
import threading
from pathlib import Path

# Fast imports first - show config immediately
import argparse

# Defer heavy imports until after config display
_heavy_imports_loaded = False
_shutdown_requested = threading.Event()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n⚠️  Shutdown requested. Finishing current operations...")
    _shutdown_requested.set()
    # Import and set the utils shutdown flag
    try:
        from utils import set_shutdown_flag
        set_shutdown_flag()
    except ImportError:
        pass
    # Force exit after a brief pause if operations don't stop
    import threading
    def force_exit():
        time.sleep(2)  # Give 2 seconds for graceful shutdown
        if _shutdown_requested.is_set():
            print("⚠️  Forcing exit...")
            os._exit(130)  # Exit code for Ctrl+C
    
    threading.Thread(target=force_exit, daemon=True).start()


# Register signal handler early
signal.signal(signal.SIGINT, signal_handler)


def load_heavy_imports():
    """Load heavy ML libraries only when needed"""
    global _heavy_imports_loaded
    if _heavy_imports_loaded:
        return
    
    from utils import setup_logging
    setup_logging()

    try:
        # These imports take time - only load when actually needed
        global chromadb, SentenceTransformer, partition, torch
        import chromadb
        from sentence_transformers import SentenceTransformer
        from unstructured.partition.auto import partition
        import torch
        _heavy_imports_loaded = True
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Install with: pip install chromadb sentence-transformers unstructured python-magic-bin torch")
        print("For GPU acceleration: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)


def get_model_shortcuts_help() -> str:
    """Generate help text for model shortcuts"""
    return """
MODEL SHORTCUTS (use with --model or -m):
═══════════════════════════════════════════

🚀 FAST & EFFICIENT (Recommended for most users):
  A, default, balanced → all-MiniLM-L6-v2 (recommended default)
    • Good balance of speed and quality
    • ~80MB model size, ~0.5s per 1000 files
    • Best for: General file search, quick results
    
  B → all-MiniLM-L12-v2 (slightly better quality)  
    • Better accuracy than A, still fast
    • ~120MB model size, ~0.7s per 1000 files
    • Best for: When you need better accuracy
    
  C → paraphrase-MiniLM-L6-v2 (alternative approach)
    • Different training approach, similar speed to A
    • ~80MB model size, ~0.5s per 1000 files
    • Best for: Alternative when A doesn't work well

⚡ SPEED FOCUSED (Ultra-fast processing):
  fast, tiny → all-MiniLM-L6-v1 (fastest)
    • Fastest possible processing
    • ~80MB model size, ~0.3s per 1000 files  
    • Best for: Large codebases, quick prototyping
    
🎯 QUALITY FOCUSED (Best accuracy, slower):
  best, quality → all-mpnet-base-v2 (highest quality)
    • Best accuracy available
    • ~420MB model size, ~2s per 1000 files
    • Best for: Critical searches, research documents
    
  large → all-roberta-large-v1 (large model)
    • Very high quality, large model
    • ~1.3GB model size, ~4s per 1000 files
    • Best for: When quality is more important than speed

🔧 SPECIALIZED (Domain-specific):
  code → microsoft/codebert-base (optimized for code)
    • Understands programming concepts better
    • ~500MB model size, ~1.5s per 1000 files
    • Best for: Source code, API docs, technical files
    
  multi → paraphrase-multilingual-MiniLM-L12-v2
    • Supports multiple languages
    • ~280MB model size, ~1s per 1000 files
    • Best for: Multi-language projects

📋 USAGE EXAMPLES:
  --model A              # Balanced (recommended)
  --model fast           # Speed priority  
  --model best           # Quality priority
  --model code           # Code-specific search
  
  # Or use full model names:
  --model "sentence-transformers/your-custom-model"

💡 SELECTION GUIDE:
  • New user? Start with: --model A
  • Large codebase? Use: --model fast  
  • Research/analysis? Use: --model best
  • Code-heavy project? Use: --model code
  • Need multilingual? Use: --model multi
"""


def create_detailed_parser():
    """Create argument parser with comprehensive help"""
    parser = argparse.ArgumentParser(
        description="""
🔍 SEMANTIC FILE SEARCH - Find files by meaning, not just keywords

This AI-powered tool searches through your files using vector embeddings to understand
context and meaning. Unlike traditional search that matches exact words, this finds
files based on what you're actually looking for, even if different words are used.

🎯 WHAT MAKES THIS DIFFERENT:
• Understands context: "authentication code" finds login.py, user_auth.js, signin.html
• Finds related concepts: "error handling" finds exception.py, try_catch.js, error_utils.c
• Works across languages: Finds similar functionality in Python, JavaScript, C++, etc.
• Semantic understanding: "machine learning" finds ML, neural networks, AI files

⚡ PERFORMANCE FEATURES:
• GPU acceleration (5-10x faster with CUDA)
• Smart caching (skip unchanged files)
• Optimized scanning (fast directory traversal) 
• Multi-threaded processing
• Memory management (handles large codebases)
""".strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{get_model_shortcuts_help()}

📚 BASIC EXAMPLES:
══════════════════
  # Simple searches (recommended starting point)
  python file_search.py "machine learning code"
  python file_search.py "database connection"
  python file_search.py "user authentication"
  python file_search.py "error handling patterns"

📊 INTERMEDIATE EXAMPLES:  
═══════════════════════════
  # Adjust number of results
  python file_search.py "API documentation" -k 5
  
  # Control result quality (0.0=loose, 1.0=exact)
  python file_search.py "neural networks" -s 0.7
  
  # Search different directory
  python file_search.py "config files" -r "C:/Projects/MyApp"
  
  # Use different AI models
  python file_search.py "database queries" --model fast    # Speed
  python file_search.py "algorithm research" --model best  # Quality
  python file_search.py "JavaScript functions" --model code # Code-optimized

🔧 ADVANCED EXAMPLES:
════════════════════
  # Specific file types only
  python file_search.py "React components" --extensions .js .jsx .ts .tsx
  
  # Large file handling
  python file_search.py "documentation" --max-size 500  # 500MB max
  
  # Performance tuning
  python file_search.py "test files" --threads 16 --gpu-batch-size 512
  
  # Development/debugging
  python file_search.py "utility functions" --debug --cache --perf-report

🎛️ PARAMETER DETAILS:
═══════════════════════

SEARCH BEHAVIOR:
  -k, --top-k N           Max results returned (1-100)
                          • Default: 10
                          • Examples: -k 5 (concise), -k 25 (comprehensive)
                          
  -s, --min-score X       Similarity threshold (0.0-1.0)  
                          • Default: 0.1 (loose matching)
                          • 0.0: Show everything (useful for debugging)
                          • 0.3: Somewhat related files
                          • 0.6: Quite similar files  
                          • 0.8: Very similar files
                          • Examples: -s 0.0 (see all), -s 0.7 (precise)

INPUT CONTROL:
  -r, --root-dir PATH     Directory to search
                          • Default: Current directory
                          • Examples: -r "C:/Code", -r "/home/user/projects"
                          
  --extensions LIST       File types to include
                          • Default: All supported types (~50 extensions)
                          • Examples: --extensions .py .js .md
                          • Tip: Use with --debug to see what's included
                          
  --max-size MB          Maximum file size in megabytes
                          • Default: 200 MB
                          • Examples: --max-size 50 (small files only)
                          • Note: Larger files take more processing time

AI MODEL SELECTION:
  -m, --model NAME       AI model for understanding files
                          • Default: A (balanced speed/quality)
                          • See model shortcuts table above
                          • Examples: --model fast, --model code
                          • Custom: --model "your/model/name"

PERFORMANCE TUNING:
  --cache                Enable smart caching
                          • Skips unchanged files in subsequent runs
                          • First run: slower, later runs: much faster
                          • Example: --cache (recommended for repeated searches)
                          
  --clear-cache          Reset cache before running
                          • Use when files changed but timestamps didn't
                          • Example: --clear-cache --cache
                          
  --threads N            Worker threads for file processing
                          • Default: 4 (auto-detected: up to CPU cores × 2)
                          • Range: 1-32 recommended
                          • Examples: --threads 1 (slow/safe), --threads 16 (fast)
                          
  --no-gpu               Disable GPU acceleration
                          • Default: Auto-detect and use GPU if available
                          • Use when GPU causes issues
                          • Example: --no-gpu (force CPU-only)
                          
  --gpu-batch-size N     GPU processing batch size
                          • Default: 256 (good for most GPUs)
                          • Lower: 64-128 (older/smaller GPUs)
                          • Higher: 512-1024 (high-end GPUs)
                          • Examples: --gpu-batch-size 128 (conservative)

DEVELOPMENT OPTIONS:
  --max-files N          Limit files processed (testing)
                          • No default (process all files)
                          • Examples: --max-files 100 (quick test)
                          
  --priority-exts        Process common files first
                          • Prioritizes .txt, .py, .js, .md, .html, .css
                          • Useful for seeing results faster
                          • Example: --priority-exts (get quick results)
                          
  --no-fast-scan         Use original scanning method
                          • Default: Fast optimized scanning enabled
                          • Use for debugging scan issues
                          • Example: --no-fast-scan (slower but thorough)
                          
  --no-persist           Don't save search index to disk
                          • Default: Save index for faster subsequent runs
                          • Use for one-time searches
                          • Example: --no-persist (temporary search)

DEBUG & ANALYSIS:
  --debug                Show detailed processing information
                          • File processing details
                          • Search score breakdown  
                          • Performance insights
                          • Example: --debug (understand what's happening)
                          
  -v, --verbose          Enable verbose logging
                          • Technical details and warnings
                          • Useful for troubleshooting
                          • Example: --verbose (technical users)
                          
  --perf-report          Show performance breakdown
                          • Time spent in each phase
                          • Bottleneck identification
                          • Example: --perf-report (optimize performance)

🚀 QUICK START GUIDE:
═══════════════════════
1. Basic search:         python file_search.py "what you're looking for"
2. Enable caching:       python file_search.py "query" --cache  
3. Faster results:       python file_search.py "query" --model fast
4. Better quality:       python file_search.py "query" --model best
5. Debug issues:         python file_search.py "query" --debug

🔧 TROUBLESHOOTING:
═══════════════════
Slow startup?           Try --model fast or install GPU support
No results?             Try --min-score 0.0 --debug to see all scores  
Memory issues?          Use --max-files 1000 --model tiny
GPU not working?        Check CUDA installation or use --no-gpu
Cache problems?         Use --clear-cache to reset

💡 PERFORMANCE TIPS:
═══════════════════
• First run builds index (slower), subsequent runs are fast
• Use --cache for repeated searches on same directory  
• GPU provides 5-10x speedup for large file collections
• Fast scan mode automatically skips build/cache directories
• Priority extensions show results faster for common file types
""".strip()
    )
    
    # Required arguments
    parser.add_argument("query", 
                       help="🔍 Search query - describe what you're looking for")
    
    # Search options
    search_group = parser.add_argument_group('🎯 Search Behavior')
    search_group.add_argument("-k", "--top-k", type=int, default=10,
                       metavar="N",
                       help="Maximum results to return (1-100, default: 10)")
    search_group.add_argument("-s", "--min-score", type=float, default=0.1,
                       metavar="X", 
                       help="Similarity threshold 0.0-1.0 (default: 0.1, lower=more results)")
    
    # Input options  
    input_group = parser.add_argument_group('📁 Input Control')
    input_group.add_argument("-r", "--root-dir", 
                       metavar="PATH",
                       help="Root directory to search (default: current directory)")
    input_group.add_argument("--extensions", nargs="+",
                       metavar="EXT",
                       help="File extensions to include (e.g., .txt .py .md .js)")
    input_group.add_argument("--max-size", type=int, default=200,
                       metavar="MB",
                       help="Maximum file size in MB (default: 200)")
    
    # AI model options
    model_group = parser.add_argument_group('🤖 AI Model Selection')
    model_group.add_argument("-m", "--model", default="A",
                       metavar="NAME",
                       help='AI model: use shortcuts (A, fast, best, code) or full name (default: A)')
    
    # Performance options
    perf_group = parser.add_argument_group('⚡ Performance Tuning')
    perf_group.add_argument("--cache", action="store_true",
                       help="Enable smart caching for faster subsequent runs")
    perf_group.add_argument("--clear-cache", action="store_true",
                       help="Clear existing cache before running")
    perf_group.add_argument("--no-persist", action="store_true",
                       help="Don't save search index to disk (temporary only)")
    perf_group.add_argument("--threads", type=int, default=4,
                       metavar="N",
                       help="Worker threads for file processing (default: auto-detect)")
    perf_group.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration even if CUDA available")
    perf_group.add_argument("--gpu-batch-size", type=int, default=256,
                       metavar="N",
                       help="GPU processing batch size (default: 256)")
    
    # Development options
    dev_group = parser.add_argument_group('🔧 Development Options')
    dev_group.add_argument("--no-fast-scan", action="store_true",
                       help="Disable optimized file scanning (use original method)")
    dev_group.add_argument("--max-files", type=int,
                       metavar="N",
                       help="Maximum files to process (for testing/limiting)")
    dev_group.add_argument("--priority-exts", action="store_true",
                       help="Process priority extensions (.txt, .py, .md) first")
    dev_group.add_argument("--perf-report", action="store_true",
                       help="Show detailed performance breakdown at end")
    
    # Debug options
    debug_group = parser.add_argument_group('🐛 Debug & Analysis')
    debug_group.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging for troubleshooting")
    debug_group.add_argument("--debug", action="store_true",
                       help="Show debug info about file processing and search results")
    
    return parser


def main():
    """Main function with fast startup and performance monitoring"""
    startup_time = time.time()
    
    # Enable immediate Ctrl+C response during heavy operations
    # signal.signal(signal.SIGINT, signal_handler)

    # Set up aggressive interrupt handling
    # def immediate_exit(signum, frame):
    #     print("\n⚠️  Immediate shutdown requested!")
    #     os._exit(130)
    
    # signal.signal(signal.SIGINT, immediate_exit)

    try:
        # Parse arguments first (fast)
        parser = create_detailed_parser()
        args = parser.parse_args()
        
        # Import configuration (fast - no heavy deps)
        from config import Config, resolve_model_name
        from utils import PerformanceMonitor
        
        # Show configuration immediately (before heavy imports)
        config = Config()
        config.apply_args(args)
        config.log_configuration(args)
        
        # Initialize performance monitoring if requested
        perf = None
        if args.perf_report:
            perf = PerformanceMonitor()
            perf.start_phase("Heavy Imports & GPU Detection")
        
        print("Loading AI libraries and detecting hardware...")
        
        # Now load heavy imports (shows progress)
        load_heavy_imports()
        
        # Import the heavy modules (after showing config)
        from file_processor import FileProcessor
        from vector_db import VectorDatabase
        from utils import Spinner
        
        init_time = time.time() - startup_time
        print(f"✅ Initialization complete ({init_time:.1f}s)")
        
        if perf:
            perf.end_phase()
            perf.start_phase("File Processing")
        
        # Validate root directory
        if not os.path.exists(config.root_dir):
            print(f"❌ Error: Root directory does not exist: {config.root_dir}")
            return 1
        
        # Initialize components
        processor = FileProcessor(config)
        vector_db = VectorDatabase(config)
        
        try:
            # Scan files with progress tracking
            spinner = Spinner("Scanning files [Ctrl+C to abort]", processor.progress_tracker)
            spinner.start()
            file_texts = processor.load_files_from_directory(config.root_dir, args.debug)
            processed, errors, skipped = processor.progress_tracker.get_counts()
            spinner.stop(f"Scan complete: {processed} processed, {skipped} skipped, {errors} errors")
            
            # Check for shutdown request
            if _shutdown_requested.is_set():
                print("⚠️  Scan interrupted by user")
                return 130
            
            if perf:
                perf.end_phase()
                perf.start_phase("AI Model Loading")
            
            # Initialize vector database
            vector_db.initialize()
            
            if perf:
                perf.end_phase()
                perf.start_phase("Building Search Index")
            
            # Check if we need to rebuild database
            need_rebuild = len(file_texts) > 0
            if not need_rebuild:
                # No new files, check if database exists
                try:
                    collection = vector_db.client.get_collection("documents")
                    if collection.count() == 0:
                        need_rebuild = True
                except:
                    need_rebuild = True
            
            if not file_texts and not need_rebuild:
                # Try to use existing database
                try:
                    vector_db.collection = vector_db.client.get_collection("documents")
                    if vector_db.collection.count() == 0:
                        print("No files found and no existing database.")
                        return 0
                    print(f"Using existing database with {vector_db.collection.count()} documents")
                except:
                    print("No files found to process and no existing database.")
                    return 0
            else:
                # Build or rebuild database with progress
                db_progress = vector_db.get_progress_tracker()
                spinner = Spinner("Building search index", db_progress)
                spinner.start()
                collection_name = vector_db.build_database(file_texts, force_rebuild=need_rebuild)
                spinner.stop(f"Search index built with {len(file_texts)} documents")
                
                # Check for shutdown request
                if _shutdown_requested.is_set():
                    return 130
            
            if perf:
                perf.end_phase()
                perf.start_phase("Search")
            
            # Perform search
            spinner = Spinner("Searching")
            spinner.start()
            matches = vector_db.search(args.query, config.top_k, config.min_score, args.debug)
            spinner.stop("Search complete")
            
            if perf:
                perf.end_phase()
            
            # Check for shutdown request
            if _shutdown_requested.is_set():
                return 130
            
            # Output results
            if matches:
                print(f"\n🎯 Found {len(matches)} relevant files for '{args.query}':")
                print("=" * 60)
                for i, (path, score) in enumerate(matches, 1):
                    # Show relative path if it's under root directory
                    try:
                        rel_path = os.path.relpath(path, config.root_dir)
                        if not rel_path.startswith('..'):
                            display_path = rel_path
                        else:
                            display_path = path
                    except:
                        display_path = path
                    
                    print(f"{i:2d}. [{score:.3f}] {display_path}")
                print("=" * 60)
            else:
                print(f"\n🔍 No files found matching '{args.query}'")
                print(f"   Minimum similarity score: {config.min_score:.2f}")
                if not args.debug:
                    print("💡 Try: --debug (see all scores) or -s 0.0 (lower threshold)")
            
            # Show performance report if requested
            if perf:
                perf.report()
            
            return 0
            
        finally:
            vector_db.cleanup()
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user.")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
