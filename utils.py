"""
Utility classes for the semantic file search tool
"""

import os
import sys
import threading
import time
import json
import logging
from typing import Dict

# Global flag for graceful shutdown
_shutdown_requested = threading.Event()

def set_shutdown_flag():
    """Set the global shutdown flag"""
    _shutdown_requested.set()

def is_shutdown_requested():
    """Check if shutdown has been requested"""
    return _shutdown_requested.is_set()


class PerformanceMonitor:
    """Monitor and report performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.phase_times = {}
        self.current_phase = None
    
    def start_phase(self, phase_name: str):
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.phase_times[phase_name] = {'start': time.time()}
    
    def end_phase(self):
        if self.current_phase:
            self.phase_times[self.current_phase]['duration'] = (
                time.time() - self.phase_times[self.current_phase]['start']
            )
            self.current_phase = None
    
    def report(self):
        total_time = time.time() - self.start_time
        print(f"\n📊 PERFORMANCE REPORT")
        print("=" * 40)
        print(f"🕐 Total time: {total_time:.2f}s")
        print()
        
        for phase, timing in self.phase_times.items():
            if 'duration' in timing:
                percentage = (timing['duration'] / total_time) * 100
                bar_length = int(percentage / 5)  # Scale to 20 chars max
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"{phase:<25} {timing['duration']:>6.2f}s [{bar}] {percentage:>5.1f}%")
        print("=" * 40)


class ProgressTracker:
    """Thread-safe progress tracker for file processing"""
    def __init__(self):
        self._lock = threading.Lock()
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0
    
    def increment_processed(self):
        with self._lock:
            self.processed_count += 1
    
    def increment_error(self):
        with self._lock:
            self.error_count += 1
    
    def increment_skipped(self):
        with self._lock:
            self.skipped_count += 1
    
    def get_counts(self):
        with self._lock:
            return self.processed_count, self.error_count, self.skipped_count


class Spinner:
    """Enhanced spinner with progress tracking and safe abort"""
    def __init__(self, message="Processing...", progress_tracker=None):
        # Use simple ASCII characters for better Windows compatibility
        if os.name == 'nt':
            self.spinner_cycle = ['|', '/', '-', '\\']
        else:
            # Try Unicode on non-Windows systems
            try:
                # Test Unicode support
                test_char = '⠋'
                test_char.encode(sys.stdout.encoding or 'utf-8')
                self.spinner_cycle = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            except (UnicodeEncodeError, LookupError):
                self.spinner_cycle = ['|', '/', '-', '\\']
        
        self.running = False
        self.message = message
        self.thread = None
        self.progress_tracker = progress_tracker

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        i = 0
        while self.running and not _shutdown_requested.is_set():
            try:
                if self.progress_tracker:
                    processed, errors, skipped = self.progress_tracker.get_counts()
                    total = processed + errors + skipped
                    if total > 0:
                        status = f"{self.message} {self.spinner_cycle[i % len(self.spinner_cycle)]} ({processed:,} processed, {skipped:,} skipped, {errors:,} errors)"
                    else:
                        status = f"{self.message} {self.spinner_cycle[i % len(self.spinner_cycle)]}"
                else:
                    status = f"{self.message} {self.spinner_cycle[i % len(self.spinner_cycle)]}"
                
                # Clear line and write status
                sys.stdout.write(f"\r{status}")
                sys.stdout.flush()
                time.sleep(0.1)
                i += 1
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception:
                # Handle console encoding issues gracefully
                sys.stdout.write(f"\r{self.message} .")
                sys.stdout.flush()

    def stop(self, success_msg="Done!"):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        try:
            # Clear the line first, then print success message
            if _shutdown_requested.is_set():
                sys.stdout.write(f"\r⚠️  Operation aborted by user{' ' * 50}\n")
            else:
                sys.stdout.write(f"\r✅ {success_msg}{' ' * 50}\n")
            sys.stdout.flush()
        except Exception:
            if _shutdown_requested.is_set():
                print(f"\n⚠️  Operation aborted by user")
            else:
                print(f"\n✅ {success_msg}")


class FileCache:
    """File modification time cache to avoid reprocessing unchanged files"""
    
    def __init__(self, cache_file: str = "file_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load cache: {e}")
        return {}
    
    def save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save cache: {e}")
    
    def is_file_modified(self, file_path: str) -> bool:
        try:
            current_mtime = os.path.getmtime(file_path)
            cached_mtime = self.cache.get(file_path)
            
            # If no cached time exists, consider it modified
            if cached_mtime is None:
                return True
            
            # Add small tolerance for floating point precision
            return current_mtime > (cached_mtime + 0.1)
        except OSError:
            return True
    
    def update_file_time(self, file_path: str):
        try:
            self.cache[file_path] = os.path.getmtime(file_path)
        except OSError as e:
            logging.warning(f"Could not update cache time for {file_path}: {e}")


def setup_logging():
    """Setup logging configuration to suppress external library noise"""
    # Configure logging - suppress all external library logs
    logging.basicConfig(level=logging.CRITICAL)
    
    # Suppress all noisy logs from dependencies
    logging.getLogger("unstructured").setLevel(logging.CRITICAL)
    logging.getLogger("unstructured.partition").setLevel(logging.CRITICAL)
    logging.getLogger("unstructured.partition.auto").setLevel(logging.CRITICAL)
    logging.getLogger("chromadb").setLevel(logging.CRITICAL)
    logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    logging.getLogger("torch").setLevel(logging.CRITICAL)
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    logging.getLogger("huggingface_hub.file_download").setLevel(logging.CRITICAL)  # ADD THIS
    logging.getLogger("tokenizers").setLevel(logging.CRITICAL)
    logging.getLogger("fsspec").setLevel(logging.CRITICAL)
    logging.getLogger("filelock").setLevel(logging.CRITICAL)
    logging.getLogger("tqdm").setLevel(logging.CRITICAL)
    
    # Suppress warnings module
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    
    # Disable tqdm progress bars globally
    import os
    os.environ['TQDM_DISABLE'] = '1'
    
    # Create our own logger for essential messages only
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Only log to file, not console (to avoid interfering with spinner)
    if not logger.handlers:
        file_handler = logging.FileHandler('file_search.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    return logger
