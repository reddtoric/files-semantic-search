"""
Configuration management for semantic file search
"""

import os
from pathlib import Path

# Default configuration values
DEFAULT_ROOT_DIR = os.getcwd()  # Use current directory as default
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 10
DEFAULT_MIN_SCORE = 0.0
DEFAULT_THREADS = 4
DEFAULT_GPU_BATCH_SIZE = 256
CACHE_FILE = "file_cache.json"

# Model shortcuts for easier command line usage
MODEL_SHORTCUTS = {
    # Fast & efficient models (recommended for most users)
    'A': 'sentence-transformers/all-MiniLM-L6-v2',      # Default, good balance
    'B': 'sentence-transformers/all-MiniLM-L12-v2',     # Slightly better quality
    'C': 'sentence-transformers/paraphrase-MiniLM-L6-v2',  # Alternative approach
    
    # Smaller/faster models
    'fast': 'sentence-transformers/all-MiniLM-L6-v1',   # Very fast
    'tiny': 'sentence-transformers/all-MiniLM-L6-v1',   # Alias for fast
    
    # Higher quality models (slower)
    'best': 'sentence-transformers/all-mpnet-base-v2',   # Best quality
    'large': 'sentence-transformers/all-roberta-large-v1',  # High quality
    'quality': 'sentence-transformers/all-mpnet-base-v2',  # Alias for best
    
    # Specialized models
    'code': 'microsoft/codebert-base',                   # Better for code files
    'multi': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # Multi-language
    
    # Aliases for convenience
    'default': 'sentence-transformers/all-MiniLM-L6-v2',
    'balanced': 'sentence-transformers/all-MiniLM-L6-v2',
}

# Comprehensive list of supported file extensions
SUPPORTED_EXTENSIONS = {
    # Text files
    '.txt', '.md', '.rst', '.log', '.cfg', '.ini', '.conf', '.yaml', '.yml', '.toml',
    # Code files
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
    '.cs', '.vb', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.clj', '.hs',
    '.lua', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd', '.r', '.m', '.pl',
    '.dart', '.elm', '.ex', '.exs', '.f90', '.f95', '.jl', '.nim', '.pas', '.pp', '.asm',
    '.s', '.sql', '.graphql', '.proto', '.thrift',
    # Web files
    '.html', '.htm', '.xml', '.css', '.scss', '.sass', '.less', '.svg', '.vue', '.svelte',
    '.json', '.jsonl', '.csv', '.tsv', '.xml', '.rss', '.atom',
    # Documentation
    '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.rtf', '.odt', '.odp', '.ods',
    # Configuration and data
    '.properties', '.env', '.gitignore', '.dockerignore', '.editorconfig', '.eslintrc',
    '.prettierrc', '.babelrc', '.webpack', '.rollup', '.vite', '.parcel'
}

# Smart directory exclusions for better performance
SMART_EXCLUDES = {
    # Version control
    '.git', '.svn', '.hg', '.bzr',
    # Dependencies
    'node_modules', '__pycache__', '.venv', 'venv', 'env',
    # Build outputs
    'dist', 'build', 'target', 'bin', 'obj', 'out',
    # IDE
    '.vscode', '.idea', '.vs',
    # OS
    '.ds_store', 'thumbs.db',
    # Temporary
    'tmp', 'temp', '.tmp', '.cache',
    # Logs
    'logs', 'log',
}


def resolve_model_name(model_input: str) -> str:
    """Resolve model shortcut to full model name"""
    if model_input in MODEL_SHORTCUTS:
        return MODEL_SHORTCUTS[model_input]
    return model_input  # Return as-is if not a shortcut


class Config:
    """Configuration management with performance optimizations"""
    
    def __init__(self):
        self.root_dir = DEFAULT_ROOT_DIR
        self.model_name = DEFAULT_MODEL
        self.top_k = DEFAULT_TOP_K
        self.min_score = DEFAULT_MIN_SCORE
        self.use_cache = True
        self.max_file_size = 200 * 1024 * 1024  # 200MB for large docs/PDFs
        self.exclude_dirs = SMART_EXCLUDES.copy()  # Use smart excludes by default
        self.include_extensions = SUPPORTED_EXTENSIONS.copy()
        self.persist_db = True  # Persist by default
        self.db_dir = ".chroma"
        self.threads = DEFAULT_THREADS
        self.use_gpu = True
        self.gpu_batch_size = DEFAULT_GPU_BATCH_SIZE
        
        # Performance tuning options
        self.fast_scan = True           # Use optimized scanning
        self.skip_empty_files = True    # Skip 0-byte files immediately
        self.max_files_limit = None     # Optional limit on total files to process
        self.priority_extensions = {    # Process these extensions first
            '.txt', '.md', '.py', '.js', '.ts', '.html', '.css'
        }
        
        # Memory optimization
        self.batch_size_factor = 4      # Multiplier for batch sizing
        self.gc_frequency = 1000        # Run garbage collection every N files
        
        # Auto-detect optimal thread count
        if os.cpu_count() and os.cpu_count() > 4:
            self.threads = min(16, os.cpu_count() * 2)
    
    def apply_args(self, args):
        """Apply command line arguments to configuration"""
        # Set root directory (with fallback to current dir)
        if hasattr(args, 'root_dir') and args.root_dir:
            self.root_dir = args.root_dir
        
        # Resolve model name
        self.model_name = resolve_model_name(args.model)
        
        # Apply other arguments
        self.top_k = args.top_k
        self.min_score = args.min_score
        self.use_cache = args.cache
        self.max_file_size = args.max_size * 1024 * 1024
        self.persist_db = not args.no_persist
        self.threads = args.threads
        self.use_gpu = not args.no_gpu
        self.gpu_batch_size = args.gpu_batch_size
        
        # Performance optimization settings
        self.fast_scan = not args.no_fast_scan
        if args.max_files:
            self.max_files_limit = args.max_files
        
        # Apply extensions filter if specified
        if args.extensions:
            self.include_extensions = set(ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                                        for ext in args.extensions)
        
        # Clear cache if requested
        if args.clear_cache and os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
    
    def log_configuration(self, args):
        """Log all configuration settings at startup"""
        print()  # Empty line at start
        print("🔍 SEMANTIC FILE SEARCH - CONFIGURATION")
        print("=" * 50)
        print(f"📝 Query: '{args.query}'")
        print(f"📁 Root Directory: {self.root_dir}")
        print(f"🤖 AI Model: {self._get_model_display_name()}")
        print(f"🎯 Max Results: {self.top_k}")
        print(f"📊 Min Score: {self.min_score:.2f}")
        print(f"📦 Max File Size: {self.max_file_size / (1024 * 1024):.0f} MB")
        print(f"⚡ Worker Threads: {self.threads}")
        print(f"🚀 GPU Acceleration: {'✅ Enabled' if self.use_gpu else '❌ Disabled'}")
        if self.use_gpu:
            print(f"   GPU Batch Size: {self.gpu_batch_size}")
        print(f"💾 Cache: {'✅ Enabled' if self.use_cache else '❌ Disabled'}")
        print(f"🗂️  Database Persistence: {'✅ Enabled' if self.persist_db else '❌ Disabled'}")
        print(f"⚡ Fast Scan: {'✅ Enabled' if self.fast_scan else '❌ Disabled'}")
        
        # Show performance optimizations in effect
        optimizations = []
        if args.debug:
            optimizations.append("🐛 Debug mode")
        if args.verbose:
            optimizations.append("📝 Verbose logging")
        if args.perf_report:
            optimizations.append("📊 Performance reporting")
        if args.priority_exts:
            optimizations.append("🚀 Priority extensions")
        if self.max_files_limit:
            optimizations.append(f"📊 File limit: {self.max_files_limit}")
        
        if optimizations:
            print(f"🔧 Active Options: {', '.join(optimizations)}")
        
        # Show file extensions (abbreviated for readability)
        if len(self.include_extensions) == len(SUPPORTED_EXTENSIONS):
            print(f"📄 File Types: All supported ({len(self.include_extensions)} extensions)")
        else:
            ext_list = sorted(self.include_extensions)
            if len(ext_list) > 10:
                displayed = ', '.join(ext_list[:10])
                print(f"📄 File Types: {displayed}... (+{len(ext_list)-10} more)")
            else:
                print(f"📄 File Types: {', '.join(ext_list)}")
        
        # Show excluded directories (abbreviated)
        if self.exclude_dirs:
            excluded_list = sorted(self.exclude_dirs)
            if len(excluded_list) > 8:
                displayed = ', '.join(excluded_list[:8])
                print(f"🚫 Excluded Dirs: {displayed}... (+{len(excluded_list)-8} more)")
            else:
                print(f"🚫 Excluded Dirs: {', '.join(excluded_list)}")
        
        print("=" * 50)
        print()
    
    def _get_model_display_name(self) -> str:
        """Get user-friendly model name for display"""
        # Find shortcut name if it exists
        for shortcut, full_name in MODEL_SHORTCUTS.items():
            if full_name == self.model_name:
                if shortcut in ['A', 'B', 'C']:
                    return f"{shortcut} ({full_name.split('/')[-1]})"
                else:
                    return f"{shortcut} ({full_name.split('/')[-1]})"
        
        # Return abbreviated full name
        if '/' in self.model_name:
            return f"Custom ({self.model_name.split('/')[-1]})"
        return self.model_name
